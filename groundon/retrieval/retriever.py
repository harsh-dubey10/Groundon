"""Retrieval pipeline: query -> candidates (vector and/or BM25) -> fusion -> rerank -> top-k.

    query ──► embed ──► FAISS top-N ─┐
          └─► BM25 top-N ────────────┴─► RRF fusion ─► cross-encoder rerank ─► top-k
"""

import numpy as np

from groundon.guard import scan_for_injection
from groundon.retrieval.embeddings import Embedder
from groundon.retrieval.keyword import KeywordIndex, reciprocal_rank_fusion
from groundon.retrieval.reranker import Reranker
from groundon.retrieval.vector_store import VectorStore
from groundon.types import RetrievalResult, RetrievedChunk


class Retriever:
    def __init__(self, store: VectorStore, keyword_index: KeywordIndex, embedder: Embedder,
                 reranker: Reranker | None, candidate_k: int = 20):
        self.store = store
        self.keyword_index = keyword_index
        self.embedder = embedder
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 4, hybrid: bool = True, rerank: bool = True) -> RetrievalResult:
        query_vec = self.embedder.embed([query])[0]

        vector_hits = self.store.search(query_vec, self.candidate_k)
        vector_ids = [cid for cid, _ in vector_hits]
        keyword_ids = [cid for cid, _ in self.keyword_index.search(query, self.candidate_k)] if hybrid else []

        candidate_ids = reciprocal_rank_fusion([vector_ids, keyword_ids]) if hybrid else vector_ids
        if not candidate_ids:
            return RetrievalResult(chunks=[], confidence=float("-inf"), confidence_type="vector")

        # Drop ids that no longer exist: BM25 can briefly hold chunks of a just-deleted document.
        chunks_by_id = self.store.get_chunks(candidate_ids)
        candidate_ids = [cid for cid in candidate_ids if cid in chunks_by_id]
        if not candidate_ids:
            return RetrievalResult(chunks=[], confidence=float("-inf"), confidence_type="vector")

        # Cosine for every candidate (BM25-only hits weren't scored by FAISS).
        cosine = (self.store.vectors_for(candidate_ids) @ query_vec).astype(float)
        candidates = [
            RetrievedChunk(
                chunk=chunks_by_id[cid],
                vector_score=float(cos),
                vector_rank=vector_ids.index(cid) + 1 if cid in vector_ids else None,
                bm25_rank=keyword_ids.index(cid) + 1 if cid in keyword_ids else None,
            )
            for cid, cos in zip(candidate_ids, cosine)
        ]

        use_rerank = rerank and self.reranker is not None
        if use_rerank:
            scores = self.reranker.score(query, [c.chunk.text for c in candidates])
            for c, s in zip(candidates, scores):
                c.rerank_score = s
            candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        elif not hybrid:
            candidates.sort(key=lambda c: c.vector_score, reverse=True)
        # (hybrid without rerank keeps RRF order)

        top = candidates[:top_k]
        for c in top:
            c.flags = scan_for_injection(c.chunk.text)

        if use_rerank:
            confidence, kind = max(c.rerank_score for c in top), "rerank"
        else:
            confidence, kind = float(np.max([c.vector_score for c in top])), "vector"
        return RetrievalResult(chunks=top, confidence=confidence, confidence_type=kind)
