"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.

Every span is always recorded locally (into the returned dict, and via `ResearchState.trace`
by callers). When `LANGSMITH_API_KEY` is configured, spans are also forwarded to LangSmith so
they show up in its trace UI. Tracing must never break the pipeline: any provider error is
swallowed and logged.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

try:
    from langsmith import Client as LangSmithClient
except ImportError:  # pragma: no cover - langsmith is an optional "llm" extra
    LangSmithClient = None  # type: ignore[assignment,misc]

_client_cache: dict[str, "LangSmithClient"] = {}


def _get_langsmith_client() -> "LangSmithClient | None":
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

    client = _get_langsmith_client()
    run_id = uuid4()
    if client is not None:
        try:
            client.create_run(
                name=name,
                run_type="chain",
                inputs=attributes or {},
                id=run_id,
                project_name=settings.langsmith_project,
                start_time=datetime.now(UTC),
            )
        except Exception as exc:  # pragma: no cover - tracing must never break the pipeline
            logger.warning("LangSmith create_run failed, continuing without tracing: %s", exc)
            client = None

    error: BaseException | None = None
    try:
        yield span
    except BaseException as exc:
        error = exc
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        if client is not None:
            try:
                client.update_run(
                    run_id,
                    outputs={"duration_seconds": span["duration_seconds"]},
                    error=repr(error) if error else None,
                    end_time=datetime.now(UTC),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("LangSmith update_run failed: %s", exc)
