import logging
from .logging_config import setup_logging
from .vector_store import VectorStore
from .constants import QDRANT_COLLECTION_MP
from .finrank import TEXT, RECORDS, METADATA, QUESTIONS, GOLDS
from .evaluation import retriever_eval, tf_idf_scores, bm25_scores, rankings_eval
from .hyst_query import HySTQuery

QDRANT_URL = "http://localhost:6333"

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Running main.py...")

    # vector_store.delete_collection()
    # vector_store.create_collection()
    # vector_store.upsert(TEXT, METADATA)

    hyst_query = HySTQuery("Qwen/Qwen2.5-3B-Instruct")

    vector_store = VectorStore(QDRANT_URL, QDRANT_COLLECTION_MP)
    N = len(QUESTIONS)
    scores = []
    for i in range(N):
        q = QUESTIONS[i] 
        logger.info(f"Question: {q}")
        hyst_result = hyst_query.query_filters(q)
        logger.info(f"QueryFilters = {hyst_result.filters}")
        retrieved = vector_store.retrieve(query=hyst_result.query,
                                          filters=hyst_result.filters,
                                          top_k=len(TEXT))
        ranking=[result.id for result in retrieved]
        scores.append(ranking)

    logger.info("Retriever Evaluation")
    evaluation = rankings_eval(scores, GOLDS[:N])
    for metric, value in evaluation.items():
    # for metric, value in retriever_eval(vector_store).items():
        if value is not None:
            logger.info(f"{metric}: {value:.1f}")
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
