import logging
import sys
from pathlib import Path
from .logging_config import setup_logging
from .vector_store import VectorStore
from .constants import QDRANT_STORAGE, QDRANT_COLLECTION_MP

FINRANK_PATH = Path(__file__).parent.parent.parent / "data/FinRank"
sys.path.insert(0, str(FINRANK_PATH/'baselines'))
from run_baselines import load_records, build_corpus

DATA_PATH = FINRANK_PATH / "FinRank.jsonl"
QDRANT_URL = "http://localhost:6333"

def build() -> tuple[list, list, dict]:
    records = load_records(DATA_PATH)
    texts, metadatas, index = build_corpus(records)
    return texts, metadatas, index

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    texts, _, _ = build()
    vector_store = VectorStore(QDRANT_URL, QDRANT_COLLECTION_MP)

    vector_store.upsert(texts)

    logger.info("DONE")
    
if __name__ == "__main__":
    main()
