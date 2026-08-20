"""Search client abstraction for ResearcherAgent."""

import json
import logging
import re
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
_CORPUS_DIR = Path(__file__).resolve().parents[3] / "ai_agent_offline_research_corpus_v2" / "topics"
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@lru_cache(maxsize=1)
def _load_corpus_documents() -> list[SourceDocument]:
    """Flatten every topic file's knowledge articles + source documents into SourceDocuments.

    Used as an offline fallback so the lab works without a live search API key.
    """

    documents: list[SourceDocument] = []
    if not _CORPUS_DIR.exists():
        return documents

    for topic_path in sorted(_CORPUS_DIR.glob("*.json")):
        try:
            data = json.loads(topic_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        topic_name = data.get("topic", {}).get("name", topic_path.stem)
        kb = data.get("knowledge_base", {})

        for article in kb.get("knowledge_articles", []):
            documents.append(
                SourceDocument(
                    title=f"{topic_name} — {article.get('title', article.get('article_id', ''))}",
                    url=None,
                    snippet=article.get("content", "")[:600],
                    metadata={
                        "source_id": article.get("article_id"),
                        "is_synthetic": False,
                        "provider": "offline_corpus",
                    },
                )
            )

        for doc in kb.get("source_documents", []):
            snippet = " ".join(doc.get("key_takeaways", [])) or doc.get("full_text", "")[:600]
            documents.append(
                SourceDocument(
                    title=doc.get("title", doc.get("document_id", "")),
                    url=doc.get("provenance_url"),
                    snippet=snippet[:600],
                    metadata={
                        "source_id": doc.get("document_id"),
                        "is_synthetic": doc.get("is_synthetic", False),
                        "provider": "offline_corpus",
                    },
                )
            )

    return documents


def _search_offline_corpus(query: str, max_results: int) -> list[SourceDocument]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[int, SourceDocument]] = []
    for doc in _load_corpus_documents():
        haystack = _tokenize(f"{doc.title} {doc.snippet}")
        score = len(query_tokens & haystack)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:max_results]]


def _search_tavily(query: str, max_results: int, api_key: str) -> list[SourceDocument]:
    payload = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        _TAVILY_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=get_settings().timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))

    return [
        SourceDocument(
            title=item.get("title", item.get("url", "Untitled")),
            url=item.get("url"),
            snippet=(item.get("content") or "")[:600],
            metadata={"score": item.get("score"), "provider": "tavily"},
        )
        for item in body.get("results", [])[:max_results]
    ]


class SearchClient:
    """Search client that prefers Tavily and falls back to an offline knowledge corpus."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        settings = get_settings()
        if settings.tavily_api_key:
            try:
                results = _search_tavily(query, max_results, settings.tavily_api_key)
                if results:
                    return results
                logger.warning(
                    "Tavily returned no results for %r; falling back to offline corpus", query
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                logger.warning("Tavily search failed (%s); falling back to offline corpus", exc)

        return _search_offline_corpus(query, max_results)
