import logging
from .logging_config import setup_logging
from .vector_store import VectorStore
from .constants import QDRANT_COLLECTION_MP
from .finrank import TEXT, RECORDS
from .evaluation import retriever_eval, tf_idf_scores, bm25_scores
from .hyst_query import HySTQuery

QDRANT_URL = "http://localhost:6333"

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Running main.py...")

    # logger.info("Finding all TICKERS")
    # tickers = set([r['ticker'] for r in RECORDS])
    # logger.info(tickers)
    #
    # logger.info("Finding all YEARS")
    # years = set([r['year'] for r in RECORDS])
    # logger.info(years)

    logger.info("Running HyST query")
    query = HySTQuery("Qwen/Qwen2.5-3B-Instruct")

    print(query.query_filters("How much revenue did Ford (F) earn in 1Q2024?"))

    # vector_store = VectorStore(QDRANT_URL, QDRANT_COLLECTION_MP)
    # vector_store.upsert(TEXT, METADATA)

    # logger.info("Retriever Evaluation")
    # for metric, value in retriever_eval(vector_store).items():
    #     if value:
    #         logger.info(f"{metric}: {value:.1f}")
    # logger.info("TF-IDF Evaluation")
    # for metric, value in tf_idf_scores().items():
    #     if value:
    #         logger.info(f"{metric}: {value:.1f}")
    # logger.info("BM25 Evaluation")
    # for metric, value in bm25_scores().items():
    #     if value:
    #         logger.info(f"{metric}: {value:.1f}")



    logger.info("DONE")
    
if __name__ == "__main__":
    main()
