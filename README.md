# langchain-opedd

**Licensed, rights-cleared content for your LangChain pipelines.** [Opedd](https://opedd.com/for-ai-agents) is the licensing rail between expert publishers and AI products — every article comes with a verifiable license key, on-chain proof (Tempo), and EU AI Act Article 53 attestation support. This package is the licensed alternative to scraping for RAG, agents, and AI search.

```bash
pip install langchain-opedd
```

## Load your licensed corpus into RAG (the 3-line on-ramp)

```python
from langchain_opedd import OpeddFeedLoader

loader = OpeddFeedLoader(access_key="ent_...")   # your Opedd enterprise access key
docs = loader.load()                              # LangChain Documents, licensing provenance in metadata
```

Every `Document.metadata` carries `id`, `title`, `source`, `publisher_id`, `published_at`, `author`, `content_hash`, and `licensed: True` — so answers in your pipeline stay attributable to licensed sources. Supports `since` (delta feeds), `max_documents`, and `lazy_load()` for streaming.

```python
# Straight into a vectorstore:
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
```

## Agent tools

```python
from langchain_opedd import (
    OpeddLookupTool,          # licensing status + price for any article URL (no auth)
    OpeddDirectoryTool,       # browse verified publishers (no auth)
    OpeddVerifyLicenseTool,   # verify a license key + on-chain proof (no auth)
    OpeddContentTool,         # retrieve licensed article text (buyer token)
)

tools = [
    OpeddLookupTool(),
    OpeddDirectoryTool(),
    OpeddVerifyLicenseTool(),
    OpeddContentTool(buyer_token="opedd_buyer_live_..."),
]
# use with create_react_agent / your agent framework of choice
```

## Where keys come from

- **No key needed** for discovery/verification tools.
- **Buyer token** (`opedd_buyer_live_*`): self-serve signup at [opedd.com](https://opedd.com) — no approval step.
- **Enterprise access key** (`ent_*`): issued with an enterprise license (bulk/metered catalog access).

## Autonomous purchasing

Deliberately **not** a LangChain tool (payment confirmation belongs in a richer protocol). Agents that buy licenses mid-task should use the Opedd MCP server — hosted at `https://mcp.opedd.com/mcp` or local via `npx opedd-mcp` — which includes `purchase_license` with Stripe support.

## Related

- [Opedd MCP server](https://github.com/Opedd/opedd-mcp) — 17 tools for MCP-native agents (Claude, Cursor, OpenAI)
- [opedd (Python SDK)](https://pypi.org/project/opedd/) — the client this package wraps
- [For AI agents](https://opedd.com/for-ai-agents) — full API documentation

MIT.
