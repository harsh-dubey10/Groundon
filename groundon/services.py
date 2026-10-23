"""Builds and holds every component once, so the API, scripts and eval share one setup."""

import logging
from dataclasses import dataclass

from groundon.agent.agent import Agent
from groundon.agent.company_db import seed_company_db
from groundon.agent.tools import CurrencyTool, SqlTool
from groundon.config import Settings
from groundon.generation.llm import GroqLLM, LLMClient, LLMError
from groundon.generation.rag import RagPipeline
from groundon.ingestion.pipeline import Ingestor
from groundon.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from groundon.retrieval.keyword import KeywordIndex
from groundon.retrieval.reranker import CrossEncoderReranker, Reranker
from groundon.retrieval.retriever import Retriever
from groundon.retrieval.vector_store import VectorStore

log = logging.getLogger("groundon")


@dataclass
class Services:
    settings: Settings
    store: VectorStore
    ingestor: Ingestor
    retriever: Retriever
    rag: RagPipeline
    agent: Agent | None  # None when no LLM is configured
    llm: LLMClient | None


def build_services(settings: Settings, *, embedder: Embedder | None = None, reranker: Reranker | None = None,
                   llm: LLMClient | None = None, load_reranker: bool = True) -> Services:
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    if reranker is None and load_reranker:
        reranker = CrossEncoderReranker(settings.reranker_model)

    store = VectorStore(settings.index_dir, embedder.dim)
    keyword_index = KeywordIndex()
    keyword_index.rebuild(store.all_chunks())
    ingestor = Ingestor(store, keyword_index, embedder, settings.chunk_size, settings.chunk_overlap)
    retriever = Retriever(store, keyword_index, embedder, reranker, candidate_k=settings.candidate_k)

    if llm is None and settings.api_keys:
        try:
            llm = GroqLLM(settings.api_keys, settings.llm_model, settings.llm_temperature, settings.llm_timeout_s)
            log.info("Groq client ready with %d API key(s), round-robin", llm.n_keys)
        except LLMError as e:
            log.warning("LLM disabled: %s", e)
    if llm is None:
        log.warning("No GROQ_API_KEY: /search works, /ask and /agent will return 503.")

    rag = RagPipeline(retriever, llm, verifier_model=settings.verifier_model,
                      min_rerank_score=settings.min_rerank_score, min_vector_score=settings.min_vector_score)

    seed_company_db(settings.company_db_path)
    agent = None
    if llm is not None:
        agent = Agent(llm, retriever, SqlTool(settings.company_db_path), CurrencyTool(settings.currency_api_url),
                      max_steps=settings.agent_max_steps, verifier_model=settings.verifier_model,
                      is_confident=rag.is_confident)

    if not settings.admin_api_key:
        log.warning("ADMIN_API_KEY not set: anyone who can reach the API can upload or delete documents.")

    if settings.auto_ingest_samples and store.size == 0 and settings.docs_dir.exists():
        results = ingestor.ingest_directory(settings.docs_dir)
        log.info("Auto-ingested %d sample documents", sum(r.status == "ingested" for r in results))

    return Services(settings, store, ingestor, retriever, rag, agent, llm)
