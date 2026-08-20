"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# Approximate USD price per 1K tokens (input, output). Free-tier models (":free" suffix,
# common on OpenRouter) are treated as zero cost. Unknown paid models report cost as None
# rather than a guessed number.
_KNOWN_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-3.5-turbo": (0.0005, 0.0015),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if model.endswith(":free"):
        return 0.0
    prices = next(
        (p for prefix, p in _KNOWN_PRICES_PER_1K.items() if model.startswith(prefix)), None
    )
    if prices is None or input_tokens is None or output_tokens is None:
        return None
    input_price, output_price = prices
    return round((input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price, 6)


@lru_cache(maxsize=4)
def _get_client(api_key: str, base_url: str | None, timeout: int) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


class LLMClient:
    """Provider-agnostic LLM client backed by any OpenAI-compatible endpoint."""

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError)),
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call the configured chat-completions endpoint and return a structured response."""

        settings = get_settings()
        if not settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not set. Add it to .env before calling LLMClient.complete."
            )

        client = _get_client(
            settings.openai_api_key, settings.openai_base_url, settings.timeout_seconds
        )
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except (APITimeoutError, RateLimitError, APIError):
            raise
        except Exception as exc:  # pragma: no cover - defensive against SDK/provider surprises
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        cost_usd = _estimate_cost(settings.openai_model, input_tokens, output_tokens)

        logger.info(
            "llm.complete model=%s input_tokens=%s output_tokens=%s cost_usd=%s",
            settings.openai_model,
            input_tokens,
            output_tokens,
            cost_usd,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
