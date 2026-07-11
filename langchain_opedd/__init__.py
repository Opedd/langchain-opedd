"""langchain-opedd — LangChain integration for Opedd licensed content.

Load your licensed corpus into RAG pipelines (OpeddFeedLoader) and give
agents licensed-content tools (lookup, directory, verify, retrieve) — every
article with a verifiable license key, on-chain proof, and EU AI Act
defensibility. The licensed alternative to scraping.
"""

from langchain_opedd.document_loaders import OpeddFeedLoader
from langchain_opedd.tools import (
    OpeddContentTool,
    OpeddDirectoryTool,
    OpeddLookupTool,
    OpeddVerifyLicenseTool,
)

__all__ = [
    "OpeddFeedLoader",
    "OpeddContentTool",
    "OpeddDirectoryTool",
    "OpeddLookupTool",
    "OpeddVerifyLicenseTool",
]
