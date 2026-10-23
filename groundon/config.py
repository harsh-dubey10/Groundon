"""Central configuration. Every value can be overridden with an env var (e.g. TOP_K=8)."""

import os
import re
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="allow" so numbered keys (GROQ_API_KEY_1, _2, ...) in .env are kept; see api_keys.
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    # --- LLM (Groq, OpenAI-compatible chat + tool calling) ---
    groq_api_key: str = ""
    groq_api_keys: str = ""  # comma-separated alternative to numbered GROQ_API_KEY_<n> variables
    llm_model: str = "openai/gpt-oss-120b"
    verifier_model: str = "openai/gpt-oss-20b"  # cheaper model for the groundedness check
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0

    # --- Local models (no API cost, documents never leave the machine for embedding) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # --- Storage ---
    data_dir: Path = Path("data")
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")
    company_db_path: Path = Path("data/company.db")
    auto_ingest_samples: bool = True

    # --- Chunking (characters) ---
    chunk_size: int = 500
    chunk_overlap: int = 100

    # --- Retrieval ---
    top_k: int = 4
    candidate_k: int = 20  # how many candidates each retriever returns before fusion/rerank
    use_hybrid: bool = True
    use_rerank: bool = True

    # --- Hallucination controls: abstain (skip the LLM) when retrieval confidence is low ---
    min_rerank_score: float = -7.5  # cross-encoder logit; calibrated on eval set: off-topic <= -8.5, answerable >= -6.8
    min_vector_score: float = 0.30  # cosine, used when reranking is off; off-topic <= 0.21, answerable >= 0.40

    # --- Access control: when set, uploading/deleting documents needs header X-API-Key: <value> ---
    admin_api_key: str = ""

    # --- Agent ---
    agent_max_steps: int = 6
    currency_api_url: str = "https://api.frankfurter.dev/v1/latest"

    @property
    def api_keys(self) -> list[str]:
        """All configured Groq keys, deduplicated, in order: GROQ_API_KEY, GROQ_API_KEYS, GROQ_API_KEY_<n>."""
        sources = {k.lower(): str(v) for k, v in (self.model_extra or {}).items()}
        sources.update({k.lower(): v for k, v in os.environ.items()})
        numbered = sorted((k for k in sources if re.fullmatch(r"groq_api_key_\d+", k)), key=lambda k: int(k.rsplit("_", 1)[1]))
        candidates = [self.groq_api_key, *self.groq_api_keys.split(","), *(sources[k] for k in numbered)]
        keys: list[str] = []
        for key in (c.strip() for c in candidates):
            if key and key not in keys:
                keys.append(key)
        return keys


@lru_cache
def get_settings() -> Settings:
    return Settings()
