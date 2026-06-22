# Section B — Retrieval pipeline

Retrieval over a Wikipedia-style corpus (`data/Wikipedia Entries/`, one JSON per page).
`run(queries)` returns, per query, a ranked list of `page_id` (most relevant first);
mean **NDCG@10** is the scored metric.

## Pipeline

The retrieval pipeline is primarily located in `main.py` and implements **Hybrid Retrieval + Reciprocal Rank Fusion (RRF) + Cross-Encoder Re-ranking**:

1. **Models** — We use `sentence-transformers/all-MiniLM-L6-v2` for generating dense embeddings, and `cross-encoder/ms-marco-MiniLM-L-6-v2` for high-accuracy re-ranking of the top candidates.
2. **Index** (`offline`) — `build_offline_index()` processes the corpus into:
   - **Dense FAISS index**: Stores chunk embeddings for efficient dense retrieval.
   - **BM25 Lexical index**: Pickled data structure for robust sparse lexical search.
   - **Chunks & Mappings**: JSON files storing chunk texts (for cross-encoder) and chunk-to-page-id mappings.
3. **Retrieve** (`main.py`) — `run(queries)` executes the following per query:
   - Retrieves top 40 candidate chunks using Dense (FAISS) search.
   - Retrieves top 40 candidate chunks using Sparse (BM25) search.
   - Merges and scores these candidates using Reciprocal Rank Fusion (RRF), taking the top 50.
   - Re-ranks the top 50 chunk candidates using the Cross-Encoder.
   - Deduplicates chunks back to unique `page_id`s, returning the top 10 most relevant pages.

## Setup

```bash
pip install -r requirements.txt
```

## Build the index (offline, not timed — run once on your machine / the VM)

Creates `artifacts/`. **Commit these files**; the grader does not rebuild the index.

```bash
python scripts/build_index.py
```

## Public self-test

Loads the submitted artifacts (no rebuild) and prints mean NDCG@10 over the public queries:

```bash
python scripts/eval_public.py
```

## Artifacts (`artifacts/`)

| file | format | description |
|------|--------|-------------|
| `faiss_index.bin`   | FAISS Index | Dense FAISS index of chunk embeddings |
| `bm25.pkl`          | Pickle      | Pickled BM25 model/index for sparse retrieval |
| `mapping.json`      | JSON        | Array mapping chunk indices to original `page_id` |
| `chunks.json`       | JSON        | Array of chunk texts used for cross-encoder re-ranking |

## Submission

Public GitHub repo with this code, the **required** `artifacts/`, and the presentation video.

**Video:** _<[Video Link](https://technion.zoom.us/rec/share/SZHcukFCpC09Mjb118iUn7VrK3HmeKoq7YzEsKiOSY1GGUOPeHiE-IK9_PPSFw.5FGFGcaBwWidpvy2?startTime=1782058229000)>_
