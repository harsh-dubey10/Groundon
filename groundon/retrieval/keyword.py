"""BM25 keyword search: catches exact terms (codes, names, numbers) that embeddings blur."""

import re
import threading

from rank_bm25 import BM25Okapi

from groundon.types import Chunk

_STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in is it its me my of on or "
    "our so than that the their them then there these they this to us was we what when where which "
    "who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class KeywordIndex:
    """In-memory BM25 rebuilt whenever the corpus changes.

    Fine for thousands of chunks; at larger scale use a search engine
    (OpenSearch/Elasticsearch) or SQLite FTS5, which implement BM25 natively.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._ids: list[int] = []

    def rebuild(self, chunks: list[Chunk]) -> None:
        with self._lock:
            self._ids = [c.id for c in chunks]
            self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        with self._lock:
            tokens = tokenize(query)
            if self._bm25 is None or not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [(self._ids[i], float(scores[i])) for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge ranked id lists: score(d) = sum over lists of 1 / (k + rank).

    Uses only ranks, so BM25 scores and cosine similarities never need to be put
    on the same scale.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
