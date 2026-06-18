import numpy as np
from typing import Dict, List

class VectorIndex:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.capacity = 10000
        self.size = 0
        self.vectors = np.empty((self.capacity, self.dim), dtype=np.float32)
        self.ids = np.empty(self.capacity, dtype=np.int64)
        self.id_to_row: Dict[int, int] = {}
        self.free_rows: List[int] = []

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        succ, fail = [], []
        for vid, vec in batch.items():
            if vid in self.id_to_row:
                fail.append(vid)
            else:
                row_idx = self.free_rows.pop() if self.free_rows else self.size
                if row_idx == self.size: self.size += 1
                if self.size >= self.capacity:
                    self.capacity *= 2
                    new_v = np.empty((self.capacity, self.dim), dtype=np.float32)
                    new_i = np.empty(self.capacity, dtype=np.int64)
                    new_v[:len(self.vectors)], new_i[:len(self.ids)] = self.vectors, self.ids
                    self.vectors, self.ids = new_v, new_i
                self.vectors[row_idx], self.ids[row_idx] = vec, vid
                self.id_to_row[vid] = row_idx
                succ.append(vid)
        return {"succeeded": succ, "failed": fail}

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        succ, fail = [], []
        for vid in ids:
            vid = int(vid)
            if vid not in self.id_to_row: fail.append(vid)
            else:
                self.free_rows.append(self.id_to_row.pop(vid))
                succ.append(vid)
        return {"succeeded": succ, "failed": fail}

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        if not self.id_to_row: return np.empty((len(queries), 0), dtype=np.int64)
        active_rows = list(self.id_to_row.values())
        active_vecs, active_ids = self.vectors[active_rows], self.ids[active_rows]
        scores = queries @ active_vecs.T
        k_eff = min(k, len(active_rows))
        topk_uns = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
        top_scores = np.take_along_axis(scores, topk_uns, axis=1)
        sorted_indices = np.argsort(-top_scores, axis=1)
        final_idx = np.take_along_axis(topk_uns, sorted_indices, axis=1)
        return active_ids[final_idx]
