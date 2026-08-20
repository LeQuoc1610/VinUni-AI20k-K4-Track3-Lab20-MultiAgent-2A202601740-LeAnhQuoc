"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.

Every span is always recorded locally (into the returned dict, and via `ResearchState.trace`
by callers). When `LANGSMITH_API_KEY` is configured, spans are also forwarded to LangSmith.
Spans nest automatically: a `trace_span` opened while another is still active becomes its
child, so one end-to-end workflow run shows up in the LangSmith UI as a single expandable
trace tree (supervisor -> researcher -> supervisor -> analyst -> ...) instead of disconnected
top-level runs. Tracing must never break the pipeline: any provider error is swallowed and
logged.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

try:
    from langsmith import Client as LangSmithClient
    from langsmith.run_trees import RunTree
except ImportError:  # pragma: no cover - langsmith is an optional "llm" extra
    LangSmithClient = None  # type: ignore[assignment,misc]
    RunTree = None  # type: ignore[assignment,misc]

_client_cache: dict[str, LangSmithClient] = {}
_current_run: ContextVar[RunTree | None] = ContextVar("_current_run", default=None)


def _get_langsmith_client() -> LangSmithClient | None:
    settings = get_settings()
    if LangSmithClient is None or not settings.langsmith_api_key:
        return None
    if settings.langsmith_api_key not in _client_cache:
        _client_cache[settings.langsmith_api_key] = LangSmithClient(
            api_key=settings.langsmith_api_key
        )
    return _client_cache[settings.langsmith_api_key]


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used throughout the workflow.

    Usage: `with trace_span("agent.researcher", {"iteration": 1}) as span: ...`
    """

    settings = get_settings()
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    run: RunTree | None = None
    token = None
    client = _get_langsmith_client()
    if client is not None and RunTree is not None:
        try:
            parent = _current_run.get()
            if parent is not None:
                run = parent.create_child(name=name, run_type="chain", inputs=attributes or {})
            else:
                run = RunTree(
                    name=name,
                    run_type="chain",
                    inputs=attributes or {},
                    project_name=settings.langsmith_project,
                    ls_client=client,
                )
            run.post()
            token = _current_run.set(run)
        except Exception as exc:  # pragma: no cover - tracing must never break the pipeline
            logger.warning("LangSmith span start failed, continuing without tracing: %s", exc)
            run = None

    error: BaseException | None = None
    try:
        yield span
    except BaseException as exc:
        error = exc
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if run is not None:
            try:
                run.end(
                    outputs={"duration_seconds": span["duration_seconds"]},
                    error=repr(error) if error else None,
                )
                run.patch()
            except Exception as exc:  # pragma: no cover
                logger.warning("LangSmith span end failed: %s", exc)
            finally:
                if token is not None:
                    _current_run.reset(token)
