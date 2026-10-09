"""Shared domain types passed between pipeline stages."""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: int  # also the FAISS vector id
    doc_id: str
    source: str  # original filename, used for citations
    page: int | None
    chunk_index: int
    text: str

    @property
    def location(self) -> str:
        return f"{self.source}, p.{self.page}" if self.page else self.source


@dataclass
class RetrievedChunk:
    chunk: Chunk
    vector_score: float  # cosine similarity to the query
    rerank_score: float | None = None
    bm25_rank: int | None = None  # 1-based rank in keyword search, None if not found there
    vector_rank: int | None = None
    flags: list[str] = field(default_factory=list)  # prompt-injection warnings


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    confidence: float  # best rerank score, or best cosine if reranking is off
    confidence_type: str  # "rerank" | "vector"
