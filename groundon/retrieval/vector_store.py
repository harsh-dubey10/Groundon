"""FAISS index for vectors + SQLite for the text and metadata behind each vector.

FAISS only stores numbers: it maps an integer id -> a float vector and can find the
nearest vectors to a query. Everything a citation needs (text, filename, page) lives
in SQLite, joined to FAISS by that same integer id.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch  # noqa: F401  isort: skip -- must load before faiss, see groundon/__init__.py
import faiss  # isort: skip
import numpy as np

from groundon.types import Chunk



def _faiss_single_threaded() -> None:
    """Keep FAISS off OpenMP threads to avoid clashing with PyTorch (see groundon/__init__.py).

    The OpenMP thread count is per calling thread, and FastAPI serves requests from a
    thread pool, so this must run before every FAISS call, not once at import.
    Exact search over a few thousand vectors takes ~1 ms single-threaded anyway.
    """
    faiss.omp_set_num_threads(1)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,   -- sha256 of file contents, so re-uploads are deduplicated
    source     TEXT NOT NULL,
    n_chunks   INTEGER NOT NULL,
    added_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,  -- == FAISS vector id
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    source      TEXT NOT NULL,
    page        INTEGER,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class VectorStore:
    def __init__(self, index_dir: Path, dim: int):
        self.index_dir = index_dir
        self.dim = dim
        index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = index_dir / "faiss.index"
        self._lock = threading.RLock()  # one SQLite connection + FAISS index shared by request threads
        self._db = sqlite3.connect(index_dir / "chunks.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

        _faiss_single_threaded()
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
        else:
            # Exact search with inner product. Vectors are L2-normalised, so IP == cosine.
            # IDMap2 lets us use our own ids and remove/reconstruct vectors by id.
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        if self._index.d != dim:
            raise ValueError(f"Index dimension {self._index.d} != embedding dimension {dim}. Delete {index_dir} and re-ingest.")

    # ---------- writes ----------

    def has_document(self, doc_id: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone() is not None

    def add_document(self, doc_id: str, source: str, chunks: list[tuple[int | None, str]], vectors: np.ndarray) -> int:
        """chunks: list of (page, text); vectors: matching (n, dim) float32 array."""
        with self._lock:
            start = (self._db.execute("SELECT COALESCE(MAX(id), -1) FROM chunks").fetchone()[0]) + 1
            ids = np.arange(start, start + len(chunks), dtype=np.int64)
            self._db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                (doc_id, source, len(chunks), datetime.now(timezone.utc).isoformat()),
            )
            self._db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                [(int(i), doc_id, source, page, n, text) for n, (i, (page, text)) in enumerate(zip(ids, chunks))],
            )
            _faiss_single_threaded()
            self._index.add_with_ids(vectors, ids)
            self._db.commit()
            self._save()
            return len(chunks)

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
            if not self.has_document(doc_id):
                return False
            _faiss_single_threaded()
            self._index.remove_ids(np.array(ids, dtype=np.int64))
            self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self._db.commit()
            self._save()
            return True

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))

    # ---------- reads ----------

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(chunk_id, cosine_score)] for the k nearest vectors."""
        if self._index.ntotal == 0:
            return []
        _faiss_single_threaded()
        with self._lock:
            scores, ids = self._index.search(query_vector.reshape(1, -1), min(k, self._index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def vectors_for(self, chunk_ids: list[int]) -> np.ndarray:
        with self._lock:
            return np.vstack([self._index.reconstruct(int(i)) for i in chunk_ids])

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        with self._lock:
            rows = self._db.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids).fetchall()
        return {row["id"]: Chunk(**dict(row)) for row in rows}

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return [Chunk(**dict(row)) for row in self._db.execute("SELECT * FROM chunks ORDER BY id")]

    def list_documents(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM documents ORDER BY added_at")]

    @property
    def size(self) -> int:
        return self._index.ntotal
