"""Command-line entrypoint for the lab starter."""

import sys
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark_suite
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

# LLM output can contain emoji/unicode punctuation that the default Windows console
# codepage (cp1252) cannot encode. Reconfigure to UTF-8 and disable Rich's legacy Win32
# console path (which bypasses this reconfiguration and crashes on the same characters).
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console(legacy_windows=False)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_baseline(request: ResearchQuery) -> ResearchState:
    """Single-agent baseline: one LLM call answers the query directly, no tools."""

    state = ResearchState(request=request)
    started = perf_counter()
    try:
        response = LLMClient().complete(
            system_prompt=(
                "You are a careful research assistant. Answer the user's research query "
                f"directly and concisely for an audience of {request.audience}. "
                "Note where your knowledge may be incomplete instead of inventing sources."
            ),
            user_prompt=request.query,
        )
    except LabError as exc:
        state.errors.append(str(exc))
        state.final_answer = f"Baseline failed: {exc}"
        return state
    latency = perf_counter() - started

    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "latency_seconds": latency,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline: one LLM call, no orchestration."""

    _init()
    state = run_baseline(_parse_query(query))
    metadata = state.agent_results[-1].metadata if state.agent_results else {}
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))
    if metadata:
        console.print(
            f"[dim]latency={metadata.get('latency_seconds', 0):.2f}s "
            f"input_tokens={metadata.get('input_tokens')} "
            f"output_tokens={metadata.get('output_tokens')} "
            f"cost_usd={metadata.get('cost_usd')}[/dim]"
        )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow: supervisor routes researcher/analyst/writer/critic."""

    _init()
    state = ResearchState(request=_parse_query(query))
    result = MultiAgentWorkflow().run(state)
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    config_path: Annotated[
        str, typer.Option("--config", help="YAML file with a benchmark.queries list")
    ] = "configs/lab_default.yaml",
    output_path: Annotated[
        str, typer.Option("--output", help="Where to write the markdown report")
    ] = "reports/benchmark_report.md",
) -> None:
    """Run the configured query set through baseline and multi-agent, write a comparison report."""

    _init()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    queries: list[str] = config["benchmark"]["queries"]

    def baseline_runner(query: str) -> ResearchState:
        return run_baseline(ResearchQuery(query=query))

    def multi_agent_runner(query: str) -> ResearchState:
        return MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=query)))

    metrics = run_benchmark_suite(
        queries, {"baseline": baseline_runner, "multi-agent": multi_agent_runner}
    )
    report = render_markdown_report(metrics)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    console.print(Panel.fit(f"Wrote {output} ({len(metrics)} rows)", title="Benchmark"))
    console.print(report)


if __name__ == "__main__":
    app()
