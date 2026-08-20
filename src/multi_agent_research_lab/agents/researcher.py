"""Researcher agent."""

from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise, cited research notes."""

    name = "researcher"

    def __init__(
        self, search_client: SearchClient | None = None, llm_client: LLMClient | None = None
    ) -> None:
        self._search = search_client or SearchClient()
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        started = perf_counter()
        request = state.request
        found = self._search.search(request.query, max_results=request.max_sources)

        existing_urls = {s.url for s in state.sources if s.url}
        existing_titles = {s.title for s in state.sources if not s.url}
        for source in found:
            if source.url and source.url in existing_urls:
                continue
            if not source.url and source.title in existing_titles:
                continue
            state.sources.append(source)

        if not state.sources:
            state.errors.append("Researcher found no sources for the query.")
            state.add_trace_event("researcher.no_sources", {"query": request.query})
            return state

        numbered_sources = "\n".join(
            f"[{i}] {s.title}\n{s.snippet}" for i, s in enumerate(state.sources, start=1)
        )
        try:
            response = self._llm.complete(
                system_prompt=(
                    "You are a research assistant. Summarize the numbered source snippets into "
                    "concise research notes as bullet points. Every factual statement must end "
                    "with the bracketed source number(s) it is drawn from, e.g. [1] or [1][3]. "
                    "Do not invent facts that are not in the snippets."
                ),
                user_prompt=f"Research query: {request.query}\n\nSources:\n{numbered_sources}",
            )
            research_notes = response.content
            metadata = {
                "latency_seconds": perf_counter() - started,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "sources_found": len(found),
            }
        except LabError as exc:
            state.errors.append(f"Researcher LLM synthesis failed, using raw snippets: {exc}")
            research_notes = "\n\n".join(
                f"[{i}] {s.title}: {s.snippet}" for i, s in enumerate(state.sources, start=1)
            )
            metadata = {
                "latency_seconds": perf_counter() - started,
                "sources_found": len(found),
                "fallback": True,
            }

        state.research_notes = research_notes
        state.agent_results.append(
            AgentResult(agent=AgentName.RESEARCHER, content=research_notes, metadata=metadata)
        )
        state.add_trace_event("researcher.completed", {"num_sources": len(state.sources)})
        return state
