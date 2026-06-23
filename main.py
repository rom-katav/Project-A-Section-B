"""Section B entry point with Hybrid Retrieval + RRF + Cross-Encoder."""
from typing import List
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

import urllib.request
import os

_model = None
_cross_encoder = None
_index = None
_mapping = None
_chunks_text = None
_bm25 = None

def _resolve_lfs_pointers(artifacts_dir: Path):
    base_url = "https://github.com/rom-katav/Project-A-Section-B/raw/master/artifacts/"
    for fname in ["faiss_index.bin", "mapping.json", "chunks.json", "bm25.pkl"]:
        fpath = artifacts_dir / fname
        if not fpath.exists():
            continue
            
        if fpath.stat().st_size < 2000:
            with open(fpath, "rb") as f:
                header = f.read(100)
            if b"version https://git-lfs.github.com/spec/v1" in header:
                print(f"Downloading real {fname} from GitHub LFS (this may take a moment)...")
                urllib.request.urlretrieve(base_url + fname, fpath)

def init_pipeline():
    global _model, _cross_encoder, _index, _mapping, _chunks_text, _bm25
    if _model is not None:
        return
        
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    
    _resolve_lfs_pointers(artifacts_dir)
    
    # Load Models
    _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Load Indices and Metadata
    _index = faiss.read_index(str(artifacts_dir / "faiss_index.bin"))
    with open(artifacts_dir / "mapping.json", "r") as f:
        _mapping = json.load(f)
    with open(artifacts_dir / "chunks.json", "r") as f:
        _chunks_text = json.load(f)
    with open(artifacts_dir / "bm25.pkl", "rb") as f:
        _bm25 = pickle.load(f)

def run(queries: List[str]) -> List[List[int]]:
    init_pipeline()
    
    # 1. Embed queries for dense search
    query_embeddings = _model.encode(queries, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    faiss_k = 40
    distances, faiss_indices = _index.search(query_embeddings, k=faiss_k)
    
    results = []
    
    for q_idx, query in enumerate(queries):
        tokenized_query = query.lower().split()
        
        # 2. BM25 Search
        bm25_k = 40
        bm25_scores = _bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_k]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, c_idx in enumerate(faiss_indices[q_idx]):
            if c_idx >= 0:
                rrf_scores[c_idx] = rrf_scores.get(c_idx, 0.0) + 1.0 / (60 + rank)
                
        for rank, c_idx in enumerate(bm25_top_indices):
            rrf_scores[c_idx] = rrf_scores.get(c_idx, 0.0) + 1.0 / (60 + rank)
            
        # Get top 50 merged candidates
        top_candidates = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:50]
        
        # 4. Cross-Encoder Re-ranking
        ce_pairs = [[query, _chunks_text[c_idx]] for c_idx in top_candidates]
        ce_scores = _cross_encoder.predict(ce_pairs)
        
        # Sort candidates by Cross-Encoder score
        ranked_candidates = [c_idx for _, c_idx in sorted(zip(ce_scores, top_candidates), reverse=True)]
        
        # 5. Deduplicate and map to Page IDs
        page_ids = []
        for idx in ranked_candidates:
            if idx >= 0 and idx < len(_mapping):
                page_id = int(_mapping[idx])
                if page_id not in page_ids:
                    page_ids.append(page_id)
        results.append(page_ids[:10])
        
    return results

def build_offline_index():
    from scripts.build_index import build_offline_index as _build
    _build()

if __name__ == "__main__":
    build_offline_index()
