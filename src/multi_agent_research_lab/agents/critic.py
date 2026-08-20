"""Critic agent: rule-based citation/quality gate over the writer's output."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import estimate_citation_coverage

_MIN_COVERAGE = 0.5
# Some models markdown-escape brackets (`\[2\]`) instead of plain `[2]`; match both.
_CITATION_RE = re.compile(r"\\?\[(\d+)\\?\]")


class CriticAgent(BaseAgent):
    """Fact-check-lite: flags missing citations and out-of-range source references."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        answer = state.final_answer or ""
        num_sources = len(state.sources)
        coverage = estimate_citation_coverage(answer, num_sources)
        cited = {int(n) for n in _CITATION_RE.findall(answer)}
        out_of_range = {n for n in cited if n < 1 or n > num_sources}

        findings = [f"Citation coverage: {coverage:.0%} of sentences cite at least one source."]
        if not answer.strip():
            findings.append("Final answer is empty.")
            state.errors.append("Critic: final answer is empty.")
        elif coverage < _MIN_COVERAGE:
            findings.append(
                f"Below target ({_MIN_COVERAGE:.0%}); "
                "consider asking the writer to cite more claims."
            )
            state.errors.append("Critic: citation coverage below target.")
        if out_of_range:
            findings.append(
                f"References source numbers with no matching source: {sorted(out_of_range)}."
            )
            state.errors.append("Critic: final answer cites out-of-range source numbers.")

        state.critic_notes = "\n".join(findings)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=state.critic_notes,
                metadata={
                    "citation_coverage": coverage,
                    "out_of_range_citations": sorted(out_of_range),
                },
            )
        )
        state.add_trace_event("critic.completed", {"citation_coverage": coverage})
        return state
