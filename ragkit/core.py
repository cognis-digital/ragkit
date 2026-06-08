"""Core RAG engine: chunking, TF-IDF indexing, cosine retrieval, answer synth.

No third-party deps. The index is a plain JSON document so it is portable,
inspectable, and trivially diffable.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

INDEX_VERSION = 2
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
    "for", "is", "are", "was", "were", "be", "been", "with", "as", "at",
    "by", "it", "this", "that", "these", "those", "from", "into", "its",
}


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def chunk_text(text: str, size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping word windows.

    Chunking on raw words (not tokens) keeps the source readable when shown
    back to the user. ``size``/``overlap`` are measured in words.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    step = size - overlap
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


@dataclass
class Document:
    doc_id: str
    path: str
    text: str


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    path: str
    text: str
    tf: Dict[str, float] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: int
    doc_id: str
    path: str
    score: float
    text: str


def ingest_paths(paths: List[str], exts: Tuple[str, ...] = (".txt", ".md")) -> List[Document]:
    """Read files (and recurse directories) into Document records."""
    docs: List[Document] = []
    seen = set()

    def add_file(fp: str) -> None:
        ap = os.path.abspath(fp)
        if ap in seen:
            return
        seen.add(ap)
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"cannot read {fp}: {exc}") from exc
        if text.strip():
            docs.append(Document(doc_id=os.path.basename(fp), path=ap, text=text))

    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for name in sorted(files):
                    if name.lower().endswith(exts):
                        add_file(os.path.join(root, name))
        elif os.path.isfile(p):
            add_file(p)
        else:
            raise FileNotFoundError(f"no such path: {p}")
    return docs


@dataclass
class RagIndex:
    chunks: List[Chunk]
    idf: Dict[str, float]
    chunk_size: int
    overlap: int

    # --- retrieval -------------------------------------------------------
    def _vector(self, tf: Dict[str, float]) -> Dict[str, float]:
        return {term: weight * self.idf.get(term, 0.0) for term, weight in tf.items()}

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_counts = Counter(q_tokens)
        total = sum(q_counts.values())
        q_tf = {t: c / total for t, c in q_counts.items()}
        q_vec = self._vector(q_tf)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0.0:
            return []
        results: List[SearchResult] = []
        for ch in self.chunks:
            d_vec = self._vector(ch.tf)
            d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
            if d_norm == 0.0:
                continue
            dot = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)
            score = dot / (q_norm * d_norm)
            if score > 0.0:
                results.append(SearchResult(
                    chunk_id=ch.chunk_id, doc_id=ch.doc_id, path=ch.path,
                    score=round(score, 6), text=ch.text,
                ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # --- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "index_version": INDEX_VERSION,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "idf": self.idf,
            "chunks": [asdict(c) for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RagIndex":
        if data.get("index_version") != INDEX_VERSION:
            raise ValueError(
                f"incompatible index_version {data.get('index_version')}, "
                f"expected {INDEX_VERSION}"
            )
        chunks = [Chunk(**c) for c in data["chunks"]]
        return cls(
            chunks=chunks,
            idf=data["idf"],
            chunk_size=data["chunk_size"],
            overlap=data["overlap"],
        )

    @property
    def stats(self) -> dict:
        docs = {c.doc_id for c in self.chunks}
        return {
            "documents": len(docs),
            "chunks": len(self.chunks),
            "vocabulary": len(self.idf),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }


def build_index(docs: List[Document], chunk_size: int = 80, overlap: int = 20) -> RagIndex:
    """Build a TF-IDF index over chunked documents."""
    chunks: List[Chunk] = []
    cid = 0
    for doc in docs:
        for piece in chunk_text(doc.text, size=chunk_size, overlap=overlap):
            chunks.append(Chunk(chunk_id=cid, doc_id=doc.doc_id, path=doc.path, text=piece))
            cid += 1

    n = len(chunks)
    df: Counter = Counter()
    tokenized: List[List[str]] = []
    for ch in chunks:
        toks = tokenize(ch.text)
        tokenized.append(toks)
        for term in set(toks):
            df[term] += 1

    # smoothed idf, always positive
    idf = {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}

    for ch, toks in zip(chunks, tokenized):
        if not toks:
            continue
        counts = Counter(toks)
        total = sum(counts.values())
        ch.tf = {t: c / total for t, c in counts.items()}

    return RagIndex(chunks=chunks, idf=idf, chunk_size=chunk_size, overlap=overlap)


def save_index(index: RagIndex, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index.to_dict(), fh, ensure_ascii=False, indent=2)


def load_index(path: str) -> RagIndex:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"index not found: {path} (run 'ragkit index' first)")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return RagIndex.from_dict(data)


def answer(index: RagIndex, query: str, top_k: int = 3) -> dict:
    """Extractive answer: stitch the highest-scoring chunks into a grounded
    response with explicit citations. No LLM required.
    """
    hits = index.search(query, top_k=top_k)
    if not hits:
        return {
            "query": query,
            "answer": "No relevant context found in the index.",
            "confidence": 0.0,
            "citations": [],
        }
    lines = []
    citations = []
    for i, h in enumerate(hits, 1):
        snippet = h.text.strip()
        if len(snippet) > 320:
            snippet = snippet[:317].rstrip() + "..."
        lines.append(f"[{i}] {snippet}")
        citations.append({
            "ref": i,
            "doc_id": h.doc_id,
            "chunk_id": h.chunk_id,
            "score": h.score,
        })
    return {
        "query": query,
        "answer": "\n".join(lines),
        "confidence": round(hits[0].score, 6),
        "citations": citations,
    }
