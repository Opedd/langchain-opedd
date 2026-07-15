"""Opedd document loader — licensed corpus → LangChain Documents.

The RAG on-ramp: an enterprise buyer with an ``ent_*`` access key loads
their entire licensed catalog into a vectorstore in a few lines, with the
licensing provenance carried on every Document's metadata.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from opedd import Opedd


class OpeddFeedLoader(BaseLoader):
    """Load a buyer's licensed Opedd catalog as LangChain ``Document``s.

    Every Document carries licensing provenance in ``metadata`` (article id,
    publisher, source URL, published_at) so downstream RAG answers stay
    attributable to licensed sources — the point of using Opedd over scraping.

    Example:
        .. code-block:: python

            from langchain_opedd import OpeddFeedLoader

            loader = OpeddFeedLoader(access_key="ent_...")
            docs = loader.load()          # or .lazy_load() for streaming

    Args:
        access_key: Opedd enterprise access key (``ent_*``), issued with an
            enterprise license at https://opedd.com.
        since: Optional ISO-8601 timestamp — only articles published after
            this instant (delta-feed polling).
        page_size: Articles per API page (max 200).
        max_documents: Optional hard cap on total documents loaded.
        base_url: Override the API base URL (default https://api.opedd.com).
    """

    def __init__(
        self,
        access_key: str,
        *,
        since: Optional[str] = None,
        page_size: int = 200,
        max_documents: Optional[int] = None,
        base_url: Optional[str] = None,
        client: Optional[Opedd] = None,
    ) -> None:
        self._client = client or Opedd(access_key=access_key, base_url=base_url)
        self._since = since
        self._page_size = min(page_size, 200)
        self._max_documents = max_documents

    def lazy_load(self) -> Iterator[Document]:
        cursor: Optional[str] = None
        yielded = 0
        while True:
            page: dict[str, Any] = self._client.feed.list(
                since=self._since, cursor=cursor, limit=self._page_size
            )
            data = page.get("data") or {}
            articles = page.get("articles") or data.get("articles") or []
            for a in articles:
                if self._max_documents is not None and yielded >= self._max_documents:
                    return
                yield Document(
                    page_content=a.get("content_body") or a.get("content") or "",
                    metadata={
                        "id": a.get("id"),
                        "title": a.get("title"),
                        "source": a.get("url") or a.get("source_url") or a.get("canonical_url"),
                        "publisher_id": a.get("publisher_id"),
                        "published_at": a.get("published_at"),
                        "author": a.get("author"),
                        "language": a.get("language"),
                        "word_count": a.get("word_count"),
                        "content_hash": a.get("content_hash"),
                        "provider": "opedd",
                        "licensed": True,
                    },
                )
                yielded += 1
            # The JSON feed returns the cursor at data.pagination.next_cursor
            # (successResponse envelope). The prior `_meta.next_cursor` read
            # exists ONLY in NDJSON mode → was always None → the loader
            # silently stopped after page 1, loading ≤page_size documents of
            # an arbitrarily large catalog. Guard against a non-advancing
            # cursor to avoid an infinite loop if the server ever repeats one.
            next_cursor = (data.get("pagination") or {}).get("next_cursor")
            if not next_cursor or not articles or next_cursor == cursor:
                return
            cursor = next_cursor
