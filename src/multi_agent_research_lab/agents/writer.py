"""Writer agent."""

from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces the final answer from research and analysis notes, with citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        started = perf_counter()
        source_list = "\n".join(
            f"[{i}] {s.title}" + (f" — {s.url}" if s.url else "")
            for i, s in enumerate(state.sources, start=1)
        )

        try:
            response = self._llm.complete(
                system_prompt=(
                    f"You are a technical writer producing a final answer for "
                    f"{state.request.audience}. Use the research notes and analysis to write a "
                    "clear, well-structured answer to the query. Cite sources inline with "
                    "bracketed numbers matching the numbered source list. End with a 'Sources' "
                    "section listing each cited number, its title, and URL if available."
                ),
                user_prompt=(
                    f"Query: {state.request.query}\n\n"
                    f"Research notes:\n{state.research_notes}\n\n"
                    f"Analysis:\n{state.analysis_notes}\n\n"
                    f"Source list:\n{source_list}"
                ),
            )
            final_answer = response.content
            metadata = {
                "latency_seconds": perf_counter() - started,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        except LabError as exc:
            state.errors.append(f"Writer LLM synthesis failed, assembling notes directly: {exc}")
            body = state.analysis_notes or state.research_notes or "No information gathered."
            final_answer = f"{body}\n\nSources:\n{source_list}" if source_list else body
            metadata = {"latency_seconds": perf_counter() - started, "fallback": True}

        state.final_answer = final_answer
        state.agent_results.append(
            AgentResult(agent=AgentName.WRITER, content=final_answer, metadata=metadata)
        )
        state.add_trace_event("writer.completed", {})
        return state
