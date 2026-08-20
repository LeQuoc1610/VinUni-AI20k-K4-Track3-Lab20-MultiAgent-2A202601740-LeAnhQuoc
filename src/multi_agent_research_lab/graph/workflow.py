"""LangGraph workflow: supervisor/researcher/analyst/writer/critic with conditional routing."""

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


def _safe_run(agent: BaseAgent, state: ResearchState) -> ResearchState:
    """Run an agent, converting failures into recorded errors instead of crashing the graph.

    The supervisor's retry/fallback policy relies on the target field staying empty when a
    stage fails, so it can re-route to the same stage (retry) or skip ahead (fallback).
    """

    try:
        with trace_span(f"agent.{agent.name}", {"iteration": state.iteration}):
            return agent.run(state)
    except LabError as exc:
        state.errors.append(f"{agent.name} failed: {exc}")
        state.add_trace_event(f"{agent.name}.failed", {"error": str(exc)})
        return state
    except Exception as exc:  # pragma: no cover - defensive against bugs in agent code
        state.errors.append(f"{agent.name} crashed unexpectedly: {exc}")
        state.add_trace_event(f"{agent.name}.crashed", {"error": str(exc)})
        return state


def _route_after_supervisor(state: ResearchState) -> str:
    return state.route_history[-1] if state.route_history else DONE


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self._compiled: (
            CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState] | None
        ) = None

    def build(self) -> CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState]:
        """Create and compile the LangGraph graph.

        Nodes: supervisor, researcher, analyst, writer, critic. The supervisor is the only
        node with outgoing conditional edges (decided by `state.route_history[-1]`); every
        worker routes back to the supervisor so it can decide the next step or stop.
        """

        graph: StateGraph[ResearchState, Any, ResearchState, ResearchState] = StateGraph(
            ResearchState
        )
        graph.add_node("supervisor", lambda state: _safe_run(SupervisorAgent(), state))
        graph.add_node("researcher", lambda state: _safe_run(ResearcherAgent(), state))
        graph.add_node("analyst", lambda state: _safe_run(AnalystAgent(), state))
        graph.add_node("writer", lambda state: _safe_run(WriterAgent(), state))
        graph.add_node("critic", lambda state: _safe_run(CriticAgent(), state))

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {
                AgentName.RESEARCHER.value: "researcher",
                AgentName.ANALYST.value: "analyst",
                AgentName.WRITER.value: "writer",
                AgentName.CRITIC.value: "critic",
                DONE: END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("critic", "supervisor")

        self._compiled = graph.compile()
        return self._compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Compile (if needed), invoke the graph, and convert the result back to ResearchState.

        The whole run is wrapped in one root `trace_span` so every per-agent span opened
        inside `_safe_run` nests under it, giving one coherent trace per end-to-end run
        instead of disconnected top-level spans in the LangSmith UI.
        """

        compiled = self._compiled or self.build()
        settings = get_settings()
        # Headroom beyond max_iterations: each logical step is a supervisor hop + a worker hop.
        recursion_limit = settings.max_iterations * 2 + 6
        with trace_span("workflow.multi_agent_run", {"query": state.request.query}):
            result = compiled.invoke(state, config={"recursion_limit": recursion_limit})
        return ResearchState.model_validate(result)
