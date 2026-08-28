import logging
from .logging_config import setup_logging
from .vector_store import VectorStore, build
from .constants import QDRANT_STORAGE


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    vector_store = VectorStore(QDRANT_STORAGE)

    text, metadata, index = build()
    print(text[0])
    print(metadata[0])
    print(list(index.values())[0])

    logger.info("DONE")
    
if __name__ == "__main__":
    main()
