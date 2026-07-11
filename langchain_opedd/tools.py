"""Opedd tools for LangChain agents — licensed content with provenance.

Discovery tools (lookup / directory / verify) need NO credentials; content
retrieval needs a buyer token. Autonomous purchasing is deliberately NOT a
LangChain tool — that flow (Stripe payment confirmation) lives in the Opedd
MCP server (`npx opedd-mcp` / https://mcp.opedd.com/mcp).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from opedd import Opedd


def _dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


# The opedd client requires a credential at init, but the discovery endpoints
# are public and the SDK attaches auth headers per-call only — this
# placeholder satisfies the constructor and never leaves the machine.
_PUBLIC_PLACEHOLDER = "public-tools-no-auth"


class _LookupInput(BaseModel):
    url: str = Field(description="The article URL to look up licensing status and pricing for")


class OpeddLookupTool(BaseTool):
    """Look up the licensing status + pricing of any article URL (no auth)."""

    name: str = "opedd_lookup_article"
    description: str = (
        "Check whether an article can be licensed via Opedd and at what price. "
        "Input: the article's URL. Returns licensing status, publisher, and pricing. "
        "Use before quoting or acquiring licensed content."
    )
    args_schema: Type[BaseModel] = _LookupInput
    _client: Opedd = PrivateAttr()

    def __init__(self, *, base_url: Optional[str] = None, client: Optional[Opedd] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client or Opedd(buyer_token=_PUBLIC_PLACEHOLDER, base_url=base_url)

    def _run(self, url: str) -> str:
        return _dumps(self._client.discovery.lookup_article(url=url))


class _DirectoryInput(BaseModel):
    limit: int = Field(default=10, description="Max publishers to return (1-50)")
    category: Optional[str] = Field(default=None, description="Optional category filter")


class OpeddDirectoryTool(BaseTool):
    """Browse Opedd's verified publisher catalog (no auth)."""

    name: str = "opedd_publisher_directory"
    description: str = (
        "Browse verified publishers on Opedd — names, categories, article counts, "
        "pricing, and sample articles. The discovery surface for licensed content."
    )
    args_schema: Type[BaseModel] = _DirectoryInput
    _client: Opedd = PrivateAttr()

    def __init__(self, *, base_url: Optional[str] = None, client: Optional[Opedd] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client or Opedd(buyer_token=_PUBLIC_PLACEHOLDER, base_url=base_url)

    def _run(self, limit: int = 10, category: Optional[str] = None) -> str:
        return _dumps(self._client.discovery.publisher_directory(limit=limit, category=category))


class _VerifyInput(BaseModel):
    key: str = Field(description="The Opedd license key to verify (e.g. OP-XXXX-XXXX)")


class OpeddVerifyLicenseTool(BaseTool):
    """Verify an Opedd license key — validity, coverage, on-chain proof (no auth)."""

    name: str = "opedd_verify_license"
    description: str = (
        "Verify an Opedd license key: validity, what article/publisher it covers, "
        "and its on-chain (Tempo) registration proof. Compliance-grade evidence."
    )
    args_schema: Type[BaseModel] = _VerifyInput
    _client: Opedd = PrivateAttr()

    def __init__(self, *, base_url: Optional[str] = None, client: Optional[Opedd] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client or Opedd(buyer_token=_PUBLIC_PLACEHOLDER, base_url=base_url)

    def _run(self, key: str) -> str:
        return _dumps(self._client.discovery.verify_license(key=key))


class _ContentInput(BaseModel):
    article_id: str = Field(description="The Opedd article UUID to retrieve licensed content for")


class OpeddContentTool(BaseTool):
    """Retrieve the full licensed body of an article (requires a buyer token)."""

    name: str = "opedd_get_content"
    description: str = (
        "Retrieve the full licensed text of an Opedd article by its UUID, including "
        "RAG metadata (author, language, word_count, content_hash, tags). Requires a "
        "buyer token (opedd_buyer_live_*) covering the article — get one at opedd.com."
    )
    args_schema: Type[BaseModel] = _ContentInput
    _client: Opedd = PrivateAttr()

    def __init__(
        self,
        buyer_token: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        client: Optional[Opedd] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if client is None and not buyer_token:
            raise ValueError("OpeddContentTool requires buyer_token (opedd_buyer_live_*)")
        self._client = client or Opedd(buyer_token=buyer_token, base_url=base_url)

    def _run(self, article_id: str) -> str:
        return _dumps(self._client.content.get(article_id))
