"""Cross-encoder reranking.

A bi-encoder (the embedding model) encodes query and chunk separately, which is fast
enough to search the whole corpus. A cross-encoder reads (query, chunk) together and
scores relevance far more accurately, but is too slow for the whole corpus, so we only
use it to re-order the ~20 candidates that first-stage retrieval returns.
"""

from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device="cpu")
        self._identity = torch.nn.Identity()  # return raw logits so thresholds are stable

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict([(query, t) for t in texts], activation_fn=self._identity, show_progress_bar=False)
        return [float(s) for s in scores]
