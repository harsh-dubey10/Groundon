"""Classic RAG: retrieve -> (abstain if weak) -> grounded prompt -> LLM -> citation/grounding checks."""

import time
from dataclasses import dataclass, field

from groundon.generation.grounding import (CitationReport, Verification, check_citations, normalize_citations,
                                           verify_groundedness)
from groundon.generation.llm import LLMClient
from groundon.generation.prompts import NOT_FOUND, RAG_SYSTEM, RAG_USER
from groundon.guard import format_document
from groundon.retrieval.retriever import Retriever
from groundon.types import RetrievedChunk


@dataclass
class Source:
    label: str
    source: str
    page: int | None
    text: str
    vector_score: float
    rerank_score: float | None
    bm25_rank: int | None
    vector_rank: int | None
    flags: list[str]


@dataclass
class RagAnswer:
    answer: str
    sources: list[Source]
    abstained: bool
    abstain_reason: str | None
    confidence: float | None
    confidence_type: str | None
    citations: CitationReport | None
    verification: Verification | None
    llm_called: bool
    timings_ms: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


def to_sources(chunks: list[RetrievedChunk], start: int = 1) -> list[Source]:
    return [
        Source(
            label=f"S{i}", source=c.chunk.source, page=c.chunk.page, text=c.chunk.text,
            vector_score=round(c.vector_score, 4),
            rerank_score=None if c.rerank_score is None else round(c.rerank_score, 4),
            bm25_rank=c.bm25_rank, vector_rank=c.vector_rank, flags=c.flags,
        )
        for i, c in enumerate(chunks, start=start)
    ]


def format_documents(sources: list[Source]) -> str:
    return "\n\n".join(
        format_document(s.label, f"{s.source}, p.{s.page}" if s.page else s.source, s.text, s.flags) for s in sources
    )


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient | None, *, verifier_model: str,
                 min_rerank_score: float, min_vector_score: float):
        self.retriever = retriever
        self.llm = llm
        self.verifier_model = verifier_model
        self.min_rerank_score = min_rerank_score
        self.min_vector_score = min_vector_score

    def is_confident(self, confidence: float, confidence_type: str) -> bool:
        threshold = self.min_rerank_score if confidence_type == "rerank" else self.min_vector_score
        return confidence >= threshold

    def answer(self, question: str, *, top_k: int = 4, hybrid: bool = True, rerank: bool = True,
               verify: bool = False) -> RagAnswer:
        timings = {}
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(question, top_k=top_k, hybrid=hybrid, rerank=rerank)
        timings["retrieval"] = round((time.perf_counter() - t0) * 1000)
        sources = to_sources(retrieval.chunks)

        # Hallucination control #1: if nothing relevant was retrieved, don't let the LLM guess.
        if not sources or not self.is_confident(retrieval.confidence, retrieval.confidence_type):
            reason = "no documents indexed" if not sources else (
                f"best {retrieval.confidence_type} score {retrieval.confidence:.3f} below threshold")
            return RagAnswer(answer=NOT_FOUND, sources=sources, abstained=True, abstain_reason=reason,
                             confidence=None if not sources else round(retrieval.confidence, 4),
                             confidence_type=retrieval.confidence_type, citations=None,
                             verification=None, llm_called=False, timings_ms=timings)

        if self.llm is None:
            raise RuntimeError("LLM not configured: set GROQ_API_KEY (retrieval-only endpoints still work).")

        documents = format_documents(sources)
        t1 = time.perf_counter()
        result = self.llm.chat([
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": RAG_USER.format(documents=documents, question=question)},
        ])
        timings["generation"] = round((time.perf_counter() - t1) * 1000)
        text = normalize_citations(result.content.strip())

        # Hallucination control #2: citations must point at chunks we actually retrieved.
        citations = check_citations(text, {s.label for s in sources})

        # Hallucination control #3 (opt-in): a second model checks each claim against the sources.
        verification = None
        if verify:
            t2 = time.perf_counter()
            verification = verify_groundedness(self.llm, self.verifier_model, documents, text)
            timings["verification"] = round((time.perf_counter() - t2) * 1000)

        timings["total"] = round((time.perf_counter() - t0) * 1000)
        # A cited answer that says part is missing is a partial answer, not an abstention.
        abstained = citations.is_abstention and not citations.has_citations
        return RagAnswer(answer=text, sources=sources, abstained=abstained, abstain_reason=None,
                         confidence=round(retrieval.confidence, 4), confidence_type=retrieval.confidence_type,
                         citations=citations, verification=verification, llm_called=True,
                         timings_ms=timings, usage=result.usage)
