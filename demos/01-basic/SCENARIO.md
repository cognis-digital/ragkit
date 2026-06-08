# Demo 01 - Basic RAG over a knowledge base

This demo shows the full RAGKIT loop on a small self-hosting handbook:
**ingest -> index -> serve (search/ask)** with zero external services.

## Input

`handbook.md` is a short ops handbook with sections on backups, the index
format, and retrieval scoring.

## Run it

```bash
# 1. Build a TF-IDF index from the handbook (writes .ragkit/index.json)
python -m ragkit index demos/01-basic/handbook.md

# 2. Inspect the index
python -m ragkit stats

# 3. Retrieve the most relevant chunks
python -m ragkit search "how often are backups taken" --top-k 3

# 4. Get a grounded, cited answer (no LLM required)
python -m ragkit ask "what scoring does retrieval use"

# Machine-readable output for piping into other tools:
python -m ragkit --format json ask "where is the index stored"
```

## Expected

- `index` reports the document/chunk/vocabulary counts.
- `search` ranks the backup-cadence chunk first for the backup query.
- `ask` returns an extractive answer citing the chunk it pulled from, with a
  cosine-similarity confidence score.
