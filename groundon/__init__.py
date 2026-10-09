"""Groundon: hallucination-resistant agentic RAG over enterprise documents."""

import os
import sys

# macOS only: PyTorch, FAISS and scikit-learn each ship their own OpenMP runtime
# (libomp.dylib), and mixing them crashes the process. Three measures together keep it
# stable (verified empirically; any one alone still crashes):
#   1. allow the duplicate runtime (below),
#   2. load PyTorch before FAISS (vector_store.py),
#   3. run FAISS single-threaded (vector_store.py).
# Linux/Docker builds share one runtime and are unaffected.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
