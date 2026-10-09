"""Opedd document loader — licensed corpus → LangChain Documents.

The RAG on-ramp: a buyer with an ``ent_*`` access key loads their licensed
catalogue into a vectorstore in a few lines, with the licensing provenance
carried on every Document's metadata.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from opedd import Opedd, OpeddAuthError, OpeddError, OpeddNotFoundError

# How a feed row's text is obtained — the backend contract in opedd-backend
# supabase/functions/enterprise-license/handlers/feed-contract.ts
# (feedContentAccess):
#   "included"              full text is in content_body (AI training and
#                           Full catalogue one-off orders)
#   "retrieval_per_article" monthly AI answers / client display: the feed is
#                           discovery-only; the text comes one article at a
#                           time from GET /content-delivery
#   "search_only"           pay per answer: no text in the feed, and
#                           /content-delivery refuses these licences
#                           (403 SEARCH_ONLY_LICENCE); the content is read by
#                           asking questions (POST /search). Named
#                           "metered_per_call" before feed schema
#                           self-serve-2026-10.
# Before 0.1.3 only "metered_per_call" was recognised, so a monthly AI answers
# feed loaded as Documents with empty page_content.
FETCHED_ACCESS = ("retrieval_per_article",)
# Rows whose text cannot be loaded at all: skipped (a Document with empty
# page_content is the bug 0.1.3 fixed), or kept as metadata-only Documents
# when allow_discovery_only=True.
SEARCH_ONLY_ACCESS = ("search_only", "metered_per_call")

# /content-delivery answers that mean "this article is not yours to embed",
# not "something is broken": skip the article, keep loading.
#   404  not found, or published before a forward-only subscription began
#   410  CONTENT_REVOKED — the publisher revoked it; copies must be deleted
#   403  ARTICLE_EXCLUDED — the publisher stopped licensing it
_SKIP_STATUSES = (404, 410)


def _error_code(err: OpeddError) -> Optional[str]:
    body = err.body if isinstance(err.body, dict) else {}
    inner = body.get("error")
    if isinstance(inner, dict) and isinstance(inner.get("code"), str):
        return inner["code"]
    code = body.get("code")
    return code if isinstance(code, str) else None


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

    Where the article text comes from depends on the licence you bought
    (each feed row says so in ``content_access``):

    - AI training and Full catalogue orders: the text is in the feed.
    - Monthly AI answers and client display: the feed lists the articles and
      the loader fetches each one's text from the content API. This needs a
      bearer token: pass ``buyer_email`` (the address on the order; the
      loader exchanges your access key for a token) or ``buyer_token``.
    - Pay per request (metered): same, but each fetch is billed and returns
      a snippet (up to 300 words or 25% of the article), not the full text.

    Articles the content API will not serve (revoked, withdrawn by the
    publisher, not found) are skipped. Documents with no text are never
    yielded for fetched articles, so nothing empty reaches a vectorstore.

    Args:
        access_key: Opedd enterprise access key (``ent_*``), issued with an
            order at https://opedd.com.
        since: Optional ISO-8601 timestamp — only articles published after
            this instant (delta-feed polling).
        page_size: Articles per API page (max 200).
        max_documents: Optional hard cap on total documents loaded.
        base_url: Override the API base URL (default https://api.opedd.com).
        buyer_email: The email on the order. Used to exchange ``access_key``
            for a bearer token when article text must be fetched.
        buyer_token: A bearer token (``opedd_buyer_live_*``) to fetch article
            text with, if you already have one.
        allow_discovery_only: ``True`` loads metadata-only Documents (empty
            ``page_content``) for feeds without text, and fetches nothing.
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
        allow_discovery_only: bool = False,
        buyer_email: Optional[str] = None,
        buyer_token: Optional[str] = None,
    ) -> None:
        self._client = client or Opedd(access_key=access_key, buyer_token=buyer_token, base_url=base_url)
        self._access_key = access_key
        self._base_url = base_url
        self._buyer_email = buyer_email
        self._buyer_token = buyer_token
        self._content_client: Optional[Opedd] = None
        self._since = since
        self._page_size = min(page_size, 200)
        self._max_documents = max_documents
        # Feeds without text: retrieval_per_article rows are fetched one by
        # one via /content-delivery; search_only rows (pay per answer) cannot
        # be loaded and are skipped. allow_discovery_only=True keeps both as
        # metadata-only Documents instead (catalogue/discovery workflows).
        self._allow_discovery_only = allow_discovery_only

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
                access = a.get("content_access")
                text: Optional[str]
                if access in SEARCH_ONLY_ACCESS and not self._allow_discovery_only:
                    continue
                if access in FETCHED_ACCESS and not self._allow_discovery_only:
                    text = self._fetch_text(a)
                    if not text:
                        continue
                else:
                    text = a.get("content_body") or a.get("content") or ""
                yield Document(
                    page_content=text,
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
                        "content_access": access,
                        "license_id": a.get("license_id"),
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

    def _content(self) -> Opedd:
        """A client holding a bearer token for /content-delivery."""
        if self._content_client is not None:
            return self._content_client
        on_client = getattr(self._client, "buyer_token", None)
        if isinstance(on_client, str) and on_client:
            self._content_client = self._client
        elif self._buyer_token:
            self._content_client = Opedd(
                access_key=self._access_key, buyer_token=self._buyer_token, base_url=self._base_url
            )
        elif self._buyer_email:
            # POST /enterprise-auth: access key + order email -> bearer token.
            self._content_client = Opedd.from_access_key(
                access_key=self._access_key, buyer_email=self._buyer_email, base_url=self._base_url
            )
        else:
            raise ValueError(
                "This feed lists articles without their text: it comes one article "
                "at a time from the content API, which needs a bearer token. Pass "
                "buyer_email (the email on the order) or buyer_token to "
                "OpeddFeedLoader, or allow_discovery_only=True to load "
                "metadata-only Documents."
            )
        return self._content_client

    def _fetch_text(self, article: dict[str, Any]) -> Optional[str]:
        article_id = article.get("id")
        if not isinstance(article_id, str) or not article_id:
            return None
        try:
            got = self._content().content.get(article_id)
        except OpeddNotFoundError:
            return None
        except OpeddAuthError as err:
            if _error_code(err) == "ARTICLE_EXCLUDED":
                return None
            raise
        except OpeddError as err:
            if err.status_code in _SKIP_STATUSES:
                return None
            raise
        content = got.get("content") if isinstance(got, dict) else None
        return content if isinstance(content, str) and content else None
