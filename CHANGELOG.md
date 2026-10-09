# Changelog

All notable changes to langchain-opedd are documented in this file.


## Unreleased

- `OpeddFeedLoader` skips pay-per-answer rows (`content_access: "search_only"`,
  formerly `"metered_per_call"`): their text is read by asking questions
  (`POST /search`), and fetching the article answers
  `403 SEARCH_ONLY_LICENCE`. Previously the loader fetched a snippet per row.
  `allow_discovery_only=True` still keeps them as metadata-only Documents.

## [0.1.3] — 2026-09-25

### Fixed

- `OpeddFeedLoader` loaded a monthly AI answers (or client display) feed as Documents with empty text. Those feeds list articles without their text (`content_access="retrieval_per_article"`); the loader now fetches each article's text from the content API (`GET /content-delivery`). Pay-per-request feeds (`"metered_per_call"`) are fetched the same way (billed per call, snippet-length). Feeds that carry the text (`"included"`: AI training and Full catalogue orders) are read directly, as before.

### Added

- `buyer_email` and `buyer_token` on `OpeddFeedLoader`: the bearer token used to fetch article text (with `buyer_email`, the access key is exchanged for one).
- `content_access` and `license_id` in every Document's metadata.
- Articles the content API will not serve (not found, revoked, withdrawn by the publisher) are skipped instead of stopping the load; fetched articles with no text are never yielded.

### Changed

- Requires `opedd>=0.3.2` (0.3.2 fixes the access-key exchange the loader relies on).
- A feed without text and no way to fetch it (no `buyer_email`/`buyer_token`) now raises with instructions for both metered and monthly feeds; previously only metered feeds raised, and monthly feeds silently produced empty Documents.

## [0.1.2]

- Metered (discovery-only) feeds raise instead of embedding empty documents.

## [0.1.1]

- Pagination and not-found fixes.
