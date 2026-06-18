# Project A: Dynamic Vector Index & End-to-End Retrieval

## System Prompt & LLM Execution Instructions
**To the AI Agent reading this:** Your objective is to implement `vector_index.py` (Section A) and the retrieval pipeline (Section B) exactly as described below. You must adhere strictly to the algorithmic logic and data structures provided. Do not deviate, do not hallucinate external packages, and do not write unstructured or overly verbose code.

---

## 🗄️ Section A: Dynamic Vector Index (`vector_index.py`)

### 1. Hard Constraints
* **Line Limit:** The `insert`, `delete`, and `search` methods must be **at most 20 physical lines each**. This is strictly enforced by the autograder.
* **Imports:** You may ONLY use `numpy` (as `np`) and standard `typing` (`Dict`, `List`). [cite_start]Do not import any other libraries[cite: 33, 34].
* [cite_start]**No Crashes:** The system must gracefully handle continuous changes and must not crash if asked to delete a non-existent ID[cite: 48].
* **Performance:** The data structure must be heavily optimized using pre-allocated `numpy` arrays. [cite_start]Dynamic array concatenation (e.g., `np.vstack` or `np.append` on every insert) is strictly forbidden due to speed constraints[cite: 104, 118]. 

### 2. Required Data Structures (Implement in `__init__`)
Initialize the following variables:
* `self.dim = int(dim)`: Vector dimensionality.
* `self.capacity = 10000`: Initial row capacity.
* `self.size = 0`: Current number of allocated rows.
* `self.vectors = np.empty((self.capacity, self.dim), dtype=np.float32)`: Pre-allocated matrix.
* `self.ids = np.empty(self.capacity, dtype=np.int64)`: Pre-allocated array for vector IDs.
* `self.id_to_row: Dict[int, int] = {}`: Maps a `vector_id` to its physical row index in the arrays.
* `self.free_rows: List[int] = []`: A list of row indices that were freed up by deletions.

### 3. Algorithmic Logic for Methods

**`insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]`**
1. Initialize `succeeded = []` and `failed = []`.
2. Loop over `vid, vec` in `batch.items()`.
3. If `vid` is already in `self.id_to_row`, append `vid` to `failed`.
4. Else, determine the `row_idx`. If `self.free_rows` has items, `row_idx = self.free_rows.pop()`. Else, `row_idx = self.size` and increment `self.size`.
5. (Crucial Step) If `self.size >= self.capacity`, you must double `self.capacity` and reallocate/copy `self.vectors` and `self.ids` into new arrays twice the size to maintain O(1) amortized insertion.
6. Store `vec` at `self.vectors[row_idx]` and `vid` at `self.ids[row_idx]`.
7. Add `self.id_to_row[vid] = row_idx` and append `vid` to `succeeded`.
8. [cite_start]Return `{"succeeded": succeeded, "failed": failed}`[cite: 41, 42, 43, 44, 45].

**`delete(self, ids: np.ndarray) -> Dict[str, List[int]]`**
1. Initialize `succeeded = []` and `failed = []`.
2. Loop over `vid` in `ids`.
3. [cite_start]If `vid` is NOT in `self.id_to_row`, append `vid` to `failed`[cite: 48].
4. Else, get `row_idx = self.id_to_row.pop(vid)`.
5. Append `row_idx` to `self.free_rows` and append `vid` to `succeeded`. *(Do not delete the row from the numpy array, it will simply be ignored during search and overwritten later)*.
6. [cite_start]Return `{"succeeded": succeeded, "failed": failed}`[cite: 49].

**`search(self, queries: np.ndarray, k: int) -> np.ndarray`**
1. If `self.id_to_row` is empty, return an empty array of shape `(len(queries), 0)`.
2. Extract the active rows: `active_rows = list(self.id_to_row.values())`.
3. Create view/slice of active data: `active_vecs = self.vectors[active_rows]` and `active_ids = self.ids[active_rows]`.
4. [cite_start]Calculate similarity using dot product: `scores = queries @ active_vecs.T`[cite: 53, 54].
5. [cite_start]Determine `k_eff = min(k, len(active_rows))`[cite: 52].
6. Find the top `k_eff` indices using `topk_unsorted = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]`.
7. Sort *only* those top indices by mapping back to scores and using `np.argsort`.
8. Map the final sorted indices to `active_ids` and return the resulting 2D ID array.

---

## 🔍 Section B: End-to-End Wikipedia Retrieval Pipeline

### 1. Hard Constraints
* [cite_start]**Model:** Must use exactly `sentence-transformers/all-MiniLM-L6-v2`[cite: 174].
* [cite_start]**Libraries:** standard library, `numpy`, `sentence-transformers`, `faiss-cpu` (or `faiss`)[cite: 175].
* [cite_start]**Goal:** Return one ranked list of up to 10 `page_id` integers per query[cite: 172, 173].

### 2. Algorithmic Logic: Offline Phase (`scripts/build_index.py`)
*This file is run once locally to generate the artifacts. Do not put execution time limit constraints here.*
1. **Load:** Initialize the MiniLM sentence transformer.
2. [cite_start]**Read:** Parse all JSON files in `data/Wikipedia Entries/`[cite: 150]. [cite_start]Extract `page_id` and `content`[cite: 156, 158, 160].
3. **Chunk:** Split `content` into strings of roughly 250-500 words. Maintain a parallel list tracking which `page_id` each chunk belongs to (e.g., `chunk_to_page_id_map`).
4. **Embed:** Pass all chunks through the MiniLM model to get a `(N, 384)` numpy array of `float32` embeddings.
5. **Index:** Create `index = faiss.IndexFlatIP(384)`. [cite_start]Add the embeddings to the index[cite: 176].
6. [cite_start]**Save:** Write the FAISS index to `artifacts/faiss_index.bin` and the `chunk_to_page_id_map` list to `artifacts/mapping.json`[cite: 176, 179].

### 3. Algorithmic Logic: Online Phase (`main.py` -> `run(queries: list[str])`)
*This function must be highly optimized and run in under 60 seconds total.*
1. **Init:** Load the MiniLM model. Load `artifacts/faiss_index.bin` into FAISS memory. Load `artifacts/mapping.json`. 
2. **Embed Queries:** Pass `queries` through the MiniLM model.
3. **Search:** Call `index.search(query_embeddings, k=30)` (fetch more than 10 to account for chunks coming from the exact same `page_id`).
4. **Deduplicate:** For each query's returned chunk indices, map them back to `page_id`s using the loaded mapping list. [cite_start]Append the `page_id` to the final results list for that query ONLY if it is not already in the list (preserve rank order, strip duplicates)[cite: 173].
5. [cite_start]**Return:** Slice the list to ensure a maximum of 10 `page_id`s per query and return the `list[list[int]]` structure[cite: 171, 172, 173].