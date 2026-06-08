"""RAGKIT - Batteries-included local RAG pipeline (ingest, index, serve).

Standard-library only. Zero install. Self-host retrieval-augmented generation
over your own documents with TF-IDF + cosine similarity scoring.
"""
from .core import (
    Document,
    Chunk,
    SearchResult,
    RagIndex,
    chunk_text,
    tokenize,
    ingest_paths,
    build_index,
    save_index,
    load_index,
    answer,
)

TOOL_NAME = "ragkit"
TOOL_VERSION = "1.0.0"

__all__ = [
    "Document",
    "Chunk",
    "SearchResult",
    "RagIndex",
    "chunk_text",
    "tokenize",
    "ingest_paths",
    "build_index",
    "save_index",
    "load_index",
    "answer",
    "TOOL_NAME",
    "TOOL_VERSION",
]
