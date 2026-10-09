"""Ingestion: file -> pages -> chunks -> embeddings -> FAISS + SQLite (+ BM25 rebuild)."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from groundon.ingestion.chunker import chunk_text
from groundon.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_file
from groundon.retrieval.embeddings import Embedder
from groundon.retrieval.keyword import KeywordIndex
from groundon.retrieval.vector_store import VectorStore


@dataclass
class IngestResult:
    doc_id: str
    source: str
    n_chunks: int
    status: str  # "ingested" | "duplicate" | "empty"


class Ingestor:
    def __init__(self, store: VectorStore, keyword_index: KeywordIndex, embedder: Embedder,
                 chunk_size: int, chunk_overlap: int):
        self.store = store
        self.keyword_index = keyword_index
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, path: Path, source_name: str | None = None) -> IngestResult:
        result = self._ingest(path, source_name or path.name)
        if result.status == "ingested":
            self.keyword_index.rebuild(self.store.all_chunks())
        return result

    def ingest_directory(self, directory: Path) -> list[IngestResult]:
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        results = [self._ingest(p, p.name) for p in files]
        self.keyword_index.rebuild(self.store.all_chunks())
        return results

    def _ingest(self, path: Path, source: str) -> IngestResult:
        doc_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if self.store.has_document(doc_id):
            return IngestResult(doc_id, source, 0, "duplicate")

        chunks: list[tuple[int | None, str]] = []
        for page in parse_file(path):
            # Chunk per page so every chunk keeps an exact page number for citations.
            chunks.extend((page.page, text) for text in chunk_text(page.text, self.chunk_size, self.chunk_overlap))
        if not chunks:
            return IngestResult(doc_id, source, 0, "empty")

        vectors = self.embedder.embed([text for _, text in chunks])
        n = self.store.add_document(doc_id, source, chunks, vectors)
        return IngestResult(doc_id, source, n, "ingested")
