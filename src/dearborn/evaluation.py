import logging
from dearborn.finrank import TEXT, GOLDS, QUESTIONS, per_record_metrics, aggregate, tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)

def rankings_eval(rankings: list[list[int]], golds: list[set[int]] = None) -> dict:
    if golds is None:
        golds = GOLDS
    rows = [per_record_metrics(np.array(r), g) for r, g in zip(rankings, golds)]
    return {k: aggregate(rows, k) for k in rows[0]}

def retriever_eval(vector_store) -> dict:
    scores = []
    for question in QUESTIONS:
        results = vector_store.retrieve(question, top_k=len(TEXT))
        scores.append([r.id for r in results])
    return rankings_eval(scores)

def scores_eval(score_matrix, golds: list[set[int]] = None) -> dict:
    if not golds:
        golds = GOLDS
    rankings = np.argsort(-score_matrix, axis=1)
    eval = [per_record_metrics(rankings[i], golds[i]) for i in range(len(golds))]
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




