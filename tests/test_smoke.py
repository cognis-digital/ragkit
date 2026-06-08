"""Smoke tests for RAGKIT. No network, no third-party deps."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ragkit import (  # noqa: E402
    TOOL_NAME,
    TOOL_VERSION,
    build_index,
    chunk_text,
    tokenize,
    ingest_paths,
    save_index,
    load_index,
    answer,
)
from ragkit.cli import main  # noqa: E402


CORPUS = {
    "backups.md": (
        "Automated backups run every six hours via the scheduler. "
        "Each backup is a full snapshot retained for thirty days."
    ),
    "scoring.md": (
        "Retrieval uses TF-IDF weighting combined with cosine similarity "
        "between the query vector and each chunk vector."
    ),
}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        for name, text in CORPUS.items():
            with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        self.index_path = os.path.join(self.tmp, "index.json")

    def test_metadata(self):
        self.assertEqual(TOOL_NAME, "ragkit")
        self.assertTrue(TOOL_VERSION)

    def test_tokenize_drops_stopwords(self):
        toks = tokenize("The backup is a full snapshot")
        self.assertNotIn("the", toks)
        self.assertIn("backup", toks)
        self.assertIn("snapshot", toks)

    def test_chunk_overlap(self):
        words = " ".join(str(i) for i in range(200))
        chunks = chunk_text(words, size=80, overlap=20)
        self.assertGreater(len(chunks), 1)
        # overlap: last 20 words of chunk0 == first 20 words of chunk1
        c0 = chunks[0].split()
        c1 = chunks[1].split()
        self.assertEqual(c0[-20:], c1[:20])

    def test_chunk_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", size=5, overlap=5)

    def test_build_and_search_ranking(self):
        docs = ingest_paths([self.tmp])
        self.assertEqual(len(docs), 2)
        idx = build_index(docs)
        hits = idx.search("how often do backups run", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, "backups.md")
        self.assertLessEqual(hits[0].score, 1.0 + 1e-9)
        self.assertGreater(hits[0].score, 0.0)

    def test_empty_query_returns_nothing(self):
        idx = build_index(ingest_paths([self.tmp]))
        self.assertEqual(idx.search("the a an of"), [])

    def test_roundtrip_persistence(self):
        idx = build_index(ingest_paths([self.tmp]))
        save_index(idx, self.index_path)
        self.assertTrue(os.path.isfile(self.index_path))
        loaded = load_index(self.index_path)
        self.assertEqual(loaded.stats["chunks"], idx.stats["chunks"])
        self.assertEqual(
            loaded.search("cosine similarity")[0].doc_id,
            "scoring.md",
        )

    def test_load_missing_index(self):
        with self.assertRaises(FileNotFoundError):
            load_index(os.path.join(self.tmp, "nope.json"))

    def test_answer_cited(self):
        idx = build_index(ingest_paths([self.tmp]))
        res = answer(idx, "what scoring is used for retrieval", top_k=2)
        self.assertTrue(res["citations"])
        self.assertEqual(res["citations"][0]["doc_id"], "scoring.md")
        self.assertGreater(res["confidence"], 0.0)

    def test_answer_no_context(self):
        idx = build_index(ingest_paths([self.tmp]))
        res = answer(idx, "zzz qqq xyzzy", top_k=2)
        self.assertEqual(res["confidence"], 0.0)
        self.assertEqual(res["citations"], [])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "doc.md")
        with open(self.src, "w", encoding="utf-8") as fh:
            fh.write(CORPUS["backups.md"])
        self.index_path = os.path.join(self.tmp, "idx.json")

    def test_index_then_ask_json(self):
        rc = main(["--format", "json", "index", self.src, "--index", self.index_path])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(self.index_path))
        rc = main(["--format", "json", "ask", "backups", "--index", self.index_path])
        self.assertEqual(rc, 0)

    def test_search_table(self):
        main(["index", self.src, "--index", self.index_path])
        rc = main(["search", "backups", "--index", self.index_path])
        self.assertEqual(rc, 0)

    def test_missing_index_nonzero(self):
        rc = main(["search", "backups", "--index", os.path.join(self.tmp, "missing.json")])
        self.assertEqual(rc, 1)

    def test_index_no_docs_nonzero(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        rc = main(["index", empty, "--index", self.index_path])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
