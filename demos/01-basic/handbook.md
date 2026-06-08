# Self-Host Operations Handbook

## Backups

Automated backups run every six hours via the scheduler. Each backup is a
full snapshot of the data directory and is retained for thirty days before
being pruned. Restores are performed with the `restore` command and always
target a fresh data directory to avoid clobbering live state.

## Index Format

The retrieval index is stored on disk as a single JSON file at
`.ragkit/index.json`. It contains the per-term inverse document frequency
table, the chunk texts, and per-chunk term-frequency vectors. Because it is
plain JSON it can be inspected, diffed, and version-controlled directly.

## Retrieval Scoring

Retrieval uses TF-IDF weighting combined with cosine similarity between the
query vector and each chunk vector. Stopwords are removed before scoring and
the inverse document frequency is smoothed so that rare terms are weighted
more heavily than common ones. The top-k chunks by cosine score are returned.

## Serving

The server exposes search and ask endpoints. Search returns ranked chunks
with scores; ask stitches the top chunks into an extractive answer with
explicit citations back to the source documents.
