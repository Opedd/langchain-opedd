"""Unit tests — mocked Opedd client, no network."""

from unittest.mock import MagicMock

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
