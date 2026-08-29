import logging
from .logging_config import setup_logging
from .vector_store import VectorStore
from .constants import QDRANT_COLLECTION_MP
from .finrank import TEXT
from .evaluation import retriever_eval, tf_idf_scores, bm25_scores

QDRANT_URL = "http://localhost:6333"

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    # vector_store = VectorStore(QDRANT_URL, QDRANT_COLLECTION_MP)
    # vector_store.upsert(TEXT)

    # logger.info("Retriever Evaluation")
    # for metric, value in retriever_eval(vector_store).items():
    #     if value:
    #         logger.info(f"{metric}: {value:.1f}")
    logger.info("TF-IDF Evaluation")
    for metric, value in tf_idf_scores().items():
        if value:
            logger.info(f"{metric}: {value:.1f}")
    logger.info("BM25 Evaluation")
    for metric, value in bm25_scores().items():
        if value:
            logger.info(f"{metric}: {value:.1f}")



    logger.info("DONE")
    
if __name__ == "__main__":
    main()
