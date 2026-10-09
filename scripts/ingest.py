"""Ingest a folder of documents from the command line.

    python scripts/ingest.py               # ingests data/docs
    python scripts/ingest.py path/to/dir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groundon.config import get_settings  # noqa: E402
from groundon.services import build_services  # noqa: E402


def main() -> None:
    settings = get_settings().model_copy(update={"auto_ingest_samples": False})
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.docs_dir
    services = build_services(settings, load_reranker=False)
    for r in services.ingestor.ingest_directory(directory):
        print(f"{r.status:9} {r.n_chunks:3} chunks  {r.source}")
    print(f"Index now holds {services.store.size} chunks.")


if __name__ == "__main__":
    main()
