"""Unit tests — mocked Opedd client, no network."""

from unittest.mock import MagicMock, patch

import pytest
from opedd import OpeddAuthError, OpeddError, OpeddNotFoundError

from langchain_core.documents import Document

from langchain_opedd import (
    OpeddContentTool,
    OpeddDirectoryTool,
    OpeddFeedLoader,
    OpeddLookupTool,
    OpeddVerifyLicenseTool,
)


def _mock_client() -> MagicMock:
    c = MagicMock()
    c.discovery.lookup_article.return_value = {"licensable": True, "price": 5}
    c.discovery.publisher_directory.return_value = {"publishers": [{"id": "p1"}]}
    c.discovery.verify_license.return_value = {"valid": True}
    c.content.get.return_value = {"id": "a1", "content": "body"}
    return c


def test_lookup_tool() -> None:
    c = _mock_client()
    out = OpeddLookupTool(client=c).run({"url": "https://x.example/a"})
    assert "licensable" in out
    c.discovery.lookup_article.assert_called_once_with(url="https://x.example/a")


def test_directory_tool() -> None:
    c = _mock_client()
    out = OpeddDirectoryTool(client=c).run({"limit": 5})
    assert "publishers" in out
    c.discovery.publisher_directory.assert_called_once_with(limit=5, category=None)


def test_verify_tool() -> None:
    c = _mock_client()
    out = OpeddVerifyLicenseTool(client=c).run({"key": "OP-TEST-0001"})
    assert "valid" in out


def test_content_tool_requires_token() -> None:
    try:
        OpeddContentTool()
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "buyer_token" in str(e)


def test_content_tool_with_client() -> None:
    c = _mock_client()
    out = OpeddContentTool(client=c).run({"article_id": "a1"})
    assert "body" in out


def test_feed_loader_pagination_and_metadata() -> None:
    c = MagicMock()
    # Real JSON feed shape: successResponse envelope with the cursor at
    # data.pagination.next_cursor (NOT _meta.next_cursor — that is NDJSON-only).
    # articles use the `url` key (backend maps url: a.source_url).
    c.feed.list.side_effect = [
        {"data": {"articles": [
            {"id": "a1", "title": "T1", "content_body": "B1", "url": "https://s/1",
             "publisher_id": "p1", "published_at": "2026-01-01", "author": "A"},
        ], "pagination": {"next_cursor": "c2"}}},
        {"data": {"articles": [
            {"id": "a2", "title": "T2", "content_body": "B2", "url": "https://s/2",
             "publisher_id": "p1", "published_at": "2026-01-02", "author": "A"},
        ], "pagination": {"next_cursor": None}}},
    ]
    docs = OpeddFeedLoader(access_key="ent_x", client=c).load()
    assert len(docs) == 2  # pagination MUST cross both pages (H1 regression)
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "B1"
    assert docs[0].metadata["licensed"] is True
    assert docs[0].metadata["provider"] == "opedd"
    assert docs[0].metadata["source"] == "https://s/1"  # url key resolves (M3)
    assert docs[1].metadata["id"] == "a2"


def test_feed_loader_max_documents() -> None:
    c = MagicMock()
    c.feed.list.return_value = {
        "data": {
            "articles": [{"id": f"a{i}", "content_body": "b"} for i in range(5)],
            "pagination": {"next_cursor": None},
        },
    }
    docs = OpeddFeedLoader(access_key="ent_x", client=c, max_documents=3).load()
    assert len(docs) == 3


def _feed(*rows: dict) -> MagicMock:
    c = MagicMock()
    c.buyer_token = None
    c.feed.list.return_value = {"data": {"articles": list(rows), "pagination": {"next_cursor": None}}}
    return c


def test_feed_loader_included_uses_content_body_without_fetching() -> None:
    """content_access='included' (AI training / Full catalogue): text is in the feed."""
    c = _feed({"id": "a1", "content_body": "Full text", "content_access": "included", "license_id": "L1"})
    docs = OpeddFeedLoader(access_key="ent_x", client=c).load()
    assert [d.page_content for d in docs] == ["Full text"]
    assert docs[0].metadata["content_access"] == "included"
    assert docs[0].metadata["license_id"] == "L1"
    c.content.get.assert_not_called()


def test_feed_loader_retrieval_per_article_fetches_each_article() -> None:
    """Monthly AI answers feed: discovery-only rows; text via /content-delivery.
    Before 0.1.3 these loaded as empty Documents."""
    c = _feed(
        {"id": "a1", "title": "T1", "content_body": None, "content_access": "retrieval_per_article"},
        {"id": "a2", "title": "T2", "content_body": None, "content_access": "retrieval_per_article"},
    )
    c.buyer_token = "opedd_buyer_live_x"
    # Real /content-delivery payload shape (content.get unwraps data): the text is `content`.
    c.content.get.side_effect = lambda aid: {"article_id": aid, "content": f"text of {aid}", "content_available": True}
    docs = OpeddFeedLoader(access_key="ent_x", client=c).load()
    assert [d.page_content for d in docs] == ["text of a1", "text of a2"]
    assert [call.args[0] for call in c.content.get.call_args_list] == ["a1", "a2"]
    assert docs[0].metadata["content_access"] == "retrieval_per_article"


def test_feed_loader_metered_fetches_per_call() -> None:
    c = _feed({"id": "a1", "content_body": None, "content_access": "metered_per_call"})
    c.buyer_token = "opedd_buyer_live_x"
    c.content.get.return_value = {"article_id": "a1", "content": "snippet"}
    docs = OpeddFeedLoader(access_key="ent_x", client=c).load()
    assert [d.page_content for d in docs] == ["snippet"]


def test_feed_loader_without_token_raises_with_instructions() -> None:
    """No bearer token and no buyer_email: refuse rather than embed empty text."""
    for access in ("retrieval_per_article", "metered_per_call"):
        c = _feed({"id": "a1", "content_body": None, "content_access": access})
        with pytest.raises(ValueError, match="buyer_email"):
            OpeddFeedLoader(access_key="ent_x", client=c).load()
        c.content.get.assert_not_called()


def test_feed_loader_buyer_email_exchanges_access_key() -> None:
    c = _feed({"id": "a1", "content_body": None, "content_access": "retrieval_per_article"})
    content_client = MagicMock()
    content_client.content.get.return_value = {"content": "text"}
    with patch("langchain_opedd.document_loaders.Opedd.from_access_key", return_value=content_client) as exch:
        docs = OpeddFeedLoader(access_key="ent_x", client=c, buyer_email="buyer@example.com").load()
    exch.assert_called_once_with(access_key="ent_x", buyer_email="buyer@example.com", base_url=None)
    assert [d.page_content for d in docs] == ["text"]


def _err(cls: type, status: int, body: dict) -> Exception:
    return cls("x", status_code=status, body=body)


def test_feed_loader_skips_articles_it_may_not_embed() -> None:
    """404 (not found / before the subscription began), 410 CONTENT_REVOKED,
    403 ARTICLE_EXCLUDED and content=None are skipped; loading continues."""
    rows = [{"id": f"a{i}", "content_body": None, "content_access": "retrieval_per_article"} for i in range(5)]
    c = _feed(*rows)
    c.buyer_token = "opedd_buyer_live_x"
    c.content.get.side_effect = [
        _err(OpeddNotFoundError, 404, {"success": False, "error": {"code": "NOT_FOUND", "message": "m"}}),
        _err(OpeddError, 410, {"success": False, "code": "CONTENT_REVOKED"}),
        _err(OpeddAuthError, 403, {"success": False, "error": {"code": "ARTICLE_EXCLUDED", "message": "m"}}),
        {"content": None, "content_available": False},
        {"content": "kept"},
    ]
    docs = OpeddFeedLoader(access_key="ent_x", client=c).load()
    assert [d.page_content for d in docs] == ["kept"]
    assert docs[0].metadata["id"] == "a4"


def test_feed_loader_raises_on_other_auth_failures() -> None:
    c = _feed({"id": "a1", "content_body": None, "content_access": "retrieval_per_article"})
    c.buyer_token = "opedd_buyer_live_x"
    c.content.get.side_effect = _err(OpeddAuthError, 401, {"success": False, "error": {"code": "UNAUTHORIZED", "message": "m"}})
    with pytest.raises(OpeddAuthError):
        OpeddFeedLoader(access_key="ent_x", client=c).load()


def test_feed_loader_metered_key_discovery_only_opt_in() -> None:
    c = MagicMock()
    c.feed.list.return_value = {
        "data": {
            "articles": [
                {"id": "a1", "title": "T", "content_body": None, "content_access": "metered_per_call"},
            ],
            "pagination": {"next_cursor": None},
        },
    }
    docs = OpeddFeedLoader(access_key="ent_filtered_x", client=c, allow_discovery_only=True).load()
    assert len(docs) == 1
    assert docs[0].page_content == ""
    c.content.get.assert_not_called()
    assert docs[0].metadata["title"] == "T"
