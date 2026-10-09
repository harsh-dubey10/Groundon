"""FastAPI app: REST endpoints + the single-page UI."""

import logging
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from groundon.config import get_settings
from groundon.generation.llm import LLMError
from groundon.ingestion.parsers import SUPPORTED_EXTENSIONS, UnsupportedFileType
from groundon.services import Services, build_services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "services"):  # tests inject their own
        app.state.services = build_services(get_settings())
    yield


app = FastAPI(title="Groundon", version="1.0.0", lifespan=lifespan,
              description="Hallucination-resistant agentic RAG over enterprise documents.")


def services(request: Request) -> Services:
    return request.app.state.services


# ---------------- request models ----------------

class RetrievalOptions(BaseModel):
    top_k: int = Field(4, ge=1, le=20)
    hybrid: bool = True
    rerank: bool = True


class SearchRequest(RetrievalOptions):
    query: str = Field(..., min_length=1, max_length=2000)


class AskRequest(RetrievalOptions):
    question: str = Field(..., min_length=1, max_length=2000)
    verify: bool = False


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    verify: bool = False


# ---------------- endpoints ----------------

@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    s = services(request)
    return {"status": "ok", "chunks_indexed": s.store.size, "documents": len(s.store.list_documents()),
            "llm_configured": s.llm is not None, "llm_model": s.settings.llm_model if s.llm else None,
            "llm_keys": getattr(s.llm, "n_keys", None)}


@app.get("/documents")
def list_documents(request: Request):
    return services(request).store.list_documents()


@app.post("/documents")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)):
    s = services(request)
    results = []
    for upload in files:
        name = Path(upload.filename or "upload").name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({"source": name, "status": "rejected", "error": f"unsupported type {ext}"})
            continue
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            results.append({"source": name, "status": "rejected", "error": "file larger than 20 MB"})
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(data)
            try:
                result = await run_in_threadpool(s.ingestor.ingest_file, path, name)
                results.append(asdict(result))
            except (UnsupportedFileType, ValueError, OSError) as e:
                results.append({"source": name, "status": "failed", "error": str(e)})
            except Exception as e:  # corrupt PDFs etc. raise library-specific errors
                results.append({"source": name, "status": "failed", "error": f"could not parse: {e}"})
    return {"results": results}


@app.delete("/documents/{doc_id}")
def delete_document(request: Request, doc_id: str):
    s = services(request)
    if not s.store.delete_document(doc_id):
        raise HTTPException(404, "document not found")
    s.ingestor.keyword_index.rebuild(s.store.all_chunks())
    return {"deleted": doc_id}


@app.post("/search")
def search(request: Request, body: SearchRequest):
    """Retrieval only, no LLM: useful for debugging what context the model would see."""
    result = services(request).retriever.retrieve(body.query, top_k=body.top_k, hybrid=body.hybrid, rerank=body.rerank)
    return {
        "confidence": result.confidence if result.chunks else None,
        "confidence_type": result.confidence_type,
        "results": [
            {"source": c.chunk.source, "page": c.chunk.page, "text": c.chunk.text, "vector_score": c.vector_score,
             "rerank_score": c.rerank_score, "bm25_rank": c.bm25_rank, "vector_rank": c.vector_rank, "flags": c.flags}
            for c in result.chunks
        ],
    }


@app.post("/ask")
def ask(request: Request, body: AskRequest):
    s = services(request)
    try:
        answer = s.rag.answer(body.question, top_k=body.top_k, hybrid=body.hybrid, rerank=body.rerank, verify=body.verify)
    except LLMError as e:
        raise HTTPException(502, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return asdict(answer)


@app.post("/agent")
def agent(request: Request, body: AgentRequest):
    s = services(request)
    if s.agent is None:
        raise HTTPException(503, "LLM not configured: set GROQ_API_KEY.")
    try:
        return asdict(s.agent.run(body.question, verify=body.verify))
    except LLMError as e:
        raise HTTPException(502, str(e))
