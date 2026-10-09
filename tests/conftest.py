"""Test doubles: a deterministic bag-of-words embedder and a scripted LLM, so tests are
fast, offline and don't need model downloads or an API key."""

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from groundon.config import Settings
from groundon.generation.llm import ChatResult, ToolCall
from groundon.services import build_services

DIM = 256


class FakeEmbedder:
    dim = DIM

    def embed(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class ScriptedLLM:
    """Returns queued responses in order and records every request."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def chat(self, messages, *, tools=None, model=None, json_mode=False):
        self.calls.append({"messages": messages, "tools": tools, "model": model, "json_mode": json_mode})
        if not self.responses:
            return ChatResult(content="(no scripted response)")
        r = self.responses.pop(0)
        return r(messages) if callable(r) else r


def tool_call(name, **arguments):
    return ChatResult(content="", tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)])


DOCS = {
    "leave.md": "Leave Policy\n\nEmployees receive 24 days of annual leave. A maximum of 8 unused days may be carried over to the next year.\n\nSick leave is 12 days per year.",
    "travel.md": "Travel Policy\n\nThe per diem in Germany is EUR 75 per day. Economy class is required for flights under 6 hours.",
    "security.txt": "IT Security\n\nPasswords must be at least 14 characters. Report phishing to security@example.com within 1 hour.",
}


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text)
    return d


@pytest.fixture
def settings(tmp_path: Path, docs_dir: Path) -> Settings:
    return Settings(
        _env_file=None,  # never read the developer's real .env (and API keys) in tests
        groq_api_key="", docs_dir=docs_dir, index_dir=tmp_path / "index", company_db_path=tmp_path / "company.db",
        chunk_size=200, chunk_overlap=40, min_vector_score=0.2, currency_api_url="http://currency.invalid",
    )


@pytest.fixture
def make_services(settings):
    def _make(llm=None):
        return build_services(settings, embedder=FakeEmbedder(), llm=llm, load_reranker=False)
    return _make
