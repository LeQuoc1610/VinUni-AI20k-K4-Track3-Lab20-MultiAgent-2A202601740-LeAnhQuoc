"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

# A stage is retried this many times (via re-routing to the same worker) before the
# supervisor gives up on it and moves the pipeline forward with whatever data exists.
_MAX_ATTEMPTS_PER_STAGE = 2

DONE = "done"


def _next_missing_stage(state: ResearchState) -> str | None:
    """Return the next route required to fill in `state`, or None if everything is set."""

    if not state.sources or not state.research_notes:
        return AgentName.RESEARCHER.value
    if not state.analysis_notes:
        return AgentName.ANALYST.value
    if not state.final_answer:
        return AgentName.WRITER.value
    if state.critic_notes is None:
        return AgentName.CRITIC.value
    return None


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Append the next route to `state.route_history` and return `state`.

        Policy: run researcher -> analyst -> writer -> critic in order, skipping any
        stage whose output is already populated. Each stage gets at most
        `_MAX_ATTEMPTS_PER_STAGE` attempts (tracked via how many times its route already
        appears in `route_history`); once exhausted the supervisor skips ahead instead of
        looping forever. `max_iterations` from settings is a hard stop regardless of state.
        """

        settings = get_settings()
        if state.iteration >= settings.max_iterations:
            if state.final_answer is None:
                state.errors.append(
                    f"Stopped after reaching max_iterations={settings.max_iterations} "
                    "without a final answer."
                )
            state.record_route(DONE)
            return state

        desired = _next_missing_stage(state)
        if desired is None:
            state.record_route(DONE)
            return state

        attempts_so_far = state.route_history.count(desired)
        if attempts_so_far >= _MAX_ATTEMPTS_PER_STAGE:
            state.errors.append(
                f"Stage '{desired}' did not produce output after {attempts_so_far} attempts; "
                "skipping ahead with partial data."
            )
            if desired in (AgentName.RESEARCHER.value, AgentName.ANALYST.value):
                # Skip straight to writer so the pipeline still produces a best-effort answer.
                fallback = (
                    AgentName.WRITER.value if state.final_answer is None else AgentName.CRITIC.value
                )
                state.record_route(fallback)
            else:
                # Writer or critic stuck: nothing further to try, stop the pipeline.
                state.record_route(DONE)
            return state

        state.record_route(desired)
        return state
