"""Analyst agent."""

from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights: claims, comparisons, reliability."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        started = perf_counter()
        source_list = "\n".join(
            f"[{i}] {s.title} "
            f"(provider={s.metadata.get('provider', 'unknown')}, "
            f"synthetic={s.metadata.get('is_synthetic', False)})"
            for i, s in enumerate(state.sources, start=1)
        )

        try:
            response = self._llm.complete(
                system_prompt=(
                    "You are a research analyst. Given research notes and a numbered source "
                    "list, produce structured analysis with exactly three sections:\n"
                    "1. Key claims (bulleted, each ending with its bracketed source number)\n"
                    "2. Points of agreement/disagreement between sources\n"
                    "3. Source reliability notes (flag any synthetic, low-confidence, or "
                    "single-sourced claims)\n"
                    "Keep citations as bracketed numbers matching the source list."
                ),
                user_prompt=(
                    f"Research notes:\n{state.research_notes}\n\nSource list:\n{source_list}"
                ),
            )
            analysis_notes = response.content
            metadata = {
                "latency_seconds": perf_counter() - started,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        except LabError as exc:
            state.errors.append(f"Analyst LLM synthesis failed, using research notes as-is: {exc}")
            analysis_notes = (
                f"Key claims (fallback, unanalyzed):\n{state.research_notes}\n\n"
                "Points of agreement/disagreement: not evaluated (LLM call failed).\n"
                "Source reliability notes: not evaluated (LLM call failed)."
            )
            metadata = {"latency_seconds": perf_counter() - started, "fallback": True}

        state.analysis_notes = analysis_notes
        state.agent_results.append(
            AgentResult(agent=AgentName.ANALYST, content=analysis_notes, metadata=metadata)
        )
        state.add_trace_event("analyst.completed", {})
        return state
