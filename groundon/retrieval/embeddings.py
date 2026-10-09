"""Local embedding model. Documents and queries MUST use the same model."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes inner product == cosine similarity.
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)
