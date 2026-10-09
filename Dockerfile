FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch (~200 MB instead of ~2 GB with CUDA). Separate layer so it is cached.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding + reranker models into the image: no download at startup,
# and the container works offline (only the LLM call needs the network).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

COPY groundon ./groundon
COPY scripts ./scripts
COPY eval ./eval
COPY data/docs ./data/docs

RUN useradd --create-home appuser && mkdir -p /app/data/index && chown -R appuser /app/data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "groundon.main:app", "--host", "0.0.0.0", "--port", "8000"]
