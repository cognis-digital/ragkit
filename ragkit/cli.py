"""RAGKIT command-line interface.

Subcommands:
    index   build a TF-IDF index from files/dirs
    search  retrieve top-k chunks for a query
    ask     extractive, cited answer for a query
    stats   show index statistics
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import TOOL_NAME, TOOL_VERSION
from .core import (
    answer,
    build_index,
    ingest_paths,
    load_index,
    save_index,
)

DEFAULT_INDEX = ".ragkit/index.json"


def _emit(payload: dict, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    # table format
    if "results" in payload:
        rows = payload["results"]
        if not rows:
            print("(no matches)")
            return
        print(f"{'score':>8}  {'doc_id':<24}  snippet")
        print("-" * 72)
        for r in rows:
            snip = r["text"].replace("\n", " ")
            if len(snip) > 60:
                snip = snip[:57] + "..."
            print(f"{r['score']:>8.4f}  {r['doc_id'][:24]:<24}  {snip}")
    elif "answer" in payload:
        print(payload["answer"])
        if payload["citations"]:
            print("\nSources:")
            for c in payload["citations"]:
                print(f"  [{c['ref']}] {c['doc_id']} (chunk {c['chunk_id']}, score {c['score']:.4f})")
        print(f"\nconfidence: {payload['confidence']:.4f}")
    else:
        for k, v in payload.items():
            print(f"{k}: {v}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=TOOL_NAME, description="Local RAG pipeline: ingest, index, serve.")
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    p.add_argument("--format", choices=("table", "json"), default="table", help="output format")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="build an index from files/directories")
    pi.add_argument("paths", nargs="+", help="files or directories to ingest")
    pi.add_argument("--index", default=DEFAULT_INDEX, help="output index path")
    pi.add_argument("--chunk-size", type=int, default=80, help="chunk size in words")
    pi.add_argument("--overlap", type=int, default=20, help="chunk overlap in words")

    ps = sub.add_parser("search", help="retrieve top-k chunks for a query")
    ps.add_argument("query")
    ps.add_argument("--index", default=DEFAULT_INDEX)
    ps.add_argument("--top-k", type=int, default=5)

    pa = sub.add_parser("ask", help="extractive cited answer for a query")
    pa.add_argument("query")
    pa.add_argument("--index", default=DEFAULT_INDEX)
    pa.add_argument("--top-k", type=int, default=3)

    pt = sub.add_parser("stats", help="show index statistics")
    pt.add_argument("--index", default=DEFAULT_INDEX)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    fmt = args.format
    try:
        if args.command == "index":
            docs = ingest_paths(args.paths)
            if not docs:
                print("error: no readable .txt/.md documents found", file=sys.stderr)
                return 1
            idx = build_index(docs, chunk_size=args.chunk_size, overlap=args.overlap)
            save_index(idx, args.index)
            _emit({"status": "ok", "index": args.index, **idx.stats}, fmt)
            return 0

        if args.command == "search":
            idx = load_index(args.index)
            hits = idx.search(args.query, top_k=args.top_k)
            _emit({"query": args.query, "results": [vars(h) for h in hits]}, fmt)
            return 0

        if args.command == "ask":
            idx = load_index(args.index)
            _emit(answer(idx, args.query, top_k=args.top_k), fmt)
            return 0

        if args.command == "stats":
            idx = load_index(args.index)
            _emit({"index": args.index, **idx.stats}, fmt)
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error("unknown command")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
