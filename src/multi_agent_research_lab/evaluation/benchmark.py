"""Benchmark harness for single-agent vs multi-agent comparisons."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Some models markdown-escape brackets (`\[2\]`) instead of plain `[2]`; match both.
_CITATION_RE = re.compile(r"\\?\[(\d+)\\?\]")


def estimate_citation_coverage(text: str, num_sources: int) -> float:
    """Proxy for "claims with a source / total claims": fraction of sentences that carry
    at least one bracketed citation marker, e.g. `... reduces overhead [2].`
    """

    if not text or num_sources <= 0:
        return 0.0
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _CITATION_RE.search(s))
    return round(cited / len(sentences), 3)


def _aggregate_cost(state: ResearchState) -> float | None:
    costs: list[float] = [
        r.metadata["cost_usd"]
        for r in state.agent_results
        if r.metadata.get("cost_usd") is not None
    ]
    return round(sum(costs), 6) if costs else None


def _estimate_quality(state: ResearchState, citation_coverage: float | None) -> float | None:
    """Cheap automated proxy (0-10), NOT a substitute for the peer-review rubric score."""

    if not state.final_answer:
        return None
    score = 5.0
    if len(state.final_answer) >= 200:
        score += 1.5
    if state.analysis_notes:
        score += 1.0
    if citation_coverage is not None:
        score += citation_coverage * 2.5
    if state.errors:
        score -= min(len(state.errors), 3) * 0.5
    return max(0.0, min(10.0, round(score, 2)))


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run one query through `runner`, measuring latency, cost, quality, citations, failure."""

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # runner itself crashed instead of degrading gracefully
        latency = perf_counter() - started
        padded_query = query if len(query) >= 5 else query.ljust(5, ".")
        placeholder = ResearchState(request=ResearchQuery(query=padded_query))
        placeholder.errors.append(str(exc))
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"runner raised: {exc}",
        )
        return placeholder, metrics

    latency = perf_counter() - started
    failed = not state.final_answer or not state.final_answer.strip()
    citation_coverage = (
        estimate_citation_coverage(state.final_answer or "", len(state.sources))
        if state.sources
        else None
    )
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_aggregate_cost(state),
        quality_score=_estimate_quality(state, citation_coverage),
        citation_coverage=citation_coverage,
        failure_rate=1.0 if failed else 0.0,
        notes="; ".join(state.errors) if state.errors else "",
    )
    return state, metrics


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def _aggregate_metrics(run_name: str, rows: list[BenchmarkMetrics]) -> BenchmarkMetrics:
    failures = sum(1 for r in rows if (r.failure_rate or 0) >= 1.0)
    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=_mean([r.latency_seconds for r in rows]) or 0.0,
        estimated_cost_usd=_mean([r.estimated_cost_usd for r in rows]),
        quality_score=_mean([r.quality_score for r in rows]),
        citation_coverage=_mean([r.citation_coverage for r in rows]),
        failure_rate=round(failures / len(rows), 3) if rows else None,
        notes=f"{failures}/{len(rows)} failed" if failures else "",
    )


def run_benchmark_suite(queries: list[str], runners: dict[str, Runner]) -> list[BenchmarkMetrics]:
    """Run every query through every named runner (e.g. {"baseline": ..., "multi-agent": ...}).

    Returns one row per (runner, query) plus one aggregate row per runner.
    """

    all_metrics: list[BenchmarkMetrics] = []
    for run_name, runner in runners.items():
        per_query: list[BenchmarkMetrics] = []
        for i, query in enumerate(queries, start=1):
            _, metrics = run_benchmark(f"{run_name} · q{i}", query, runner)
            per_query.append(metrics)
        all_metrics.extend(per_query)
        all_metrics.append(_aggregate_metrics(f"{run_name} (avg, n={len(per_query)})", per_query))
    return all_metrics
