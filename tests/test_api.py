import io

import pytest
from fastapi.testclient import TestClient

from groundon.generation.llm import ChatResult
from groundon.main import app
from tests.conftest import ScriptedLLM


@pytest.fixture
def client(make_services):
    def _client(llm=None):
        app.state.services = make_services(llm)
        return TestClient(app)
    yield _client
    del app.state.services


def test_health_and_documents(client):
    c = client()
    health = c.get("/health").json()
    assert health["documents"] == 3 and health["llm_configured"] is False
    assert len(c.get("/documents").json()) == 3
    assert "Groundon" in c.get("/").text


def test_search_needs_no_llm(client):
    r = client().post("/search", json={"query": "per diem Germany", "top_k": 2, "rerank": False})
    assert r.status_code == 200
    assert r.json()["results"][0]["source"] == "travel.md"


def test_ask_without_llm_is_503(client):
    r = client().post("/ask", json={"question": "per diem Germany", "rerank": False})
    assert r.status_code == 503
    assert client().post("/agent", json={"question": "hi"}).status_code == 503


def test_ask_returns_answer_and_sources(client):
    c = client(ScriptedLLM([ChatResult(content="EUR 75 per day [S1].")]))
    body = c.post("/ask", json={"question": "per diem Germany", "rerank": False}).json()
    assert body["answer"] == "EUR 75 per day [S1]."
    assert body["citations"]["cited"] == ["S1"] and body["sources"][0]["source"] == "travel.md"


def test_upload_and_delete(client):
    c = client()
    files = [("files", ("new.txt", io.BytesIO(b"The cafeteria opens at 8am."), "text/plain")),
             ("files", ("bad.exe", io.BytesIO(b"MZ"), "application/octet-stream"))]
    results = c.post("/documents", files=files).json()["results"]
    assert results[0]["status"] == "ingested" and results[1]["status"] == "rejected"
    assert c.post("/search", json={"query": "cafeteria opens", "rerank": False}).json()["results"][0]["source"] == "new.txt"

    assert c.delete(f"/documents/{results[0]['doc_id']}").status_code == 200
    assert c.delete("/documents/doesnotexist").status_code == 404


def test_validation(client):
    assert client().post("/ask", json={"question": ""}).status_code == 422
    assert client().post("/search", json={"query": "x", "top_k": 99}).status_code == 422
