import logging
from dearborn.finrank import TEXT, GOLDS, QUESTIONS, per_record_metrics, aggregate, tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)

def retriever_eval(vector_store) -> dict:
    scores = []
    for i, question in enumerate(QUESTIONS):
        results = vector_store.retrieve(question, top_k=len(TEXT))
        ranking = [result.id for result in results]
        scores.append(ranking)
    return scores_eval(scores)

def scores_eval(score_matrix) -> dict:
    rankings = np.argsort(-score_matrix, axis=1)
    eval = [per_record_metrics(rankings[i], GOLDS[i]) for i in range(len(GOLDS))]
    return {k: aggregate(eval, k) for k in eval[0]}

def tf_idf_scores() -> dict:
    vec = TfidfVectorizer(sublinear_tf=True)
    tf_idf = vec.fit_transform(TEXT)
    score_matrix = (vec.transform(QUESTIONS) @ tf_idf.T).toarray()
    return scores_eval(score_matrix)

def bm25_scores() -> dict:
    bm25 = BM25Okapi([tokenize(t) for t in TEXT])
    score_matrix = np.stack([bm25.get_scores(tokenize(q)) for q in QUESTIONS])
    return scores_eval(score_matrix)


