from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, UpdateResult, SearchParams, Filter, FieldCondition, MatchValue, MatchAny, Range
from .query_filters import QueryFilters 
import logging
logger = logging.getLogger(__name__)

MODEL = "sentence-transformers/all-mpnet-base-v2"

class VectorStore():
    def __init__(self, url: str, collection: str, delete: bool = False):
        logger.info(f"VectorStore for collection={collection} initialized")
        self.collection = collection
        self.client = QdrantClient(url=url)
        self.encoder = SentenceTransformer(MODEL)
        self.create_collection(delete)

    def create_collection(self, delete: bool):
        if self.client.collection_exists(self.collection):
            logger.info(f"Collection={self.collection} already exists")
            if delete:
                self.client.delete_collection(self.collection)
                logger.info(f"Collection={self.collection} deleted")
            return
        self.client.create_collection(
          collection_name=self.collection,
          vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        logger.info(f"Collection={self.collection} created.")

    def retrieve(self, query: str, filter: dict, top_k: int = 10) -> list[ScoredPoint]:
        query_filter = Filter(
            must= [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
        )
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            search_params=SearchParams(exact=True)
        ).points

        return results

    def upsert(self, texts: list[str], meta: list[tuple], delete: bool = False) -> None:
        info = self.client.get_collection(self.collection)
        if info.points_count == 5230:
            logger.info(f"Collection={self.collection} already populated; skipping upsert.")
            return
        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        points = [
            PointStruct(id=i,
                        vector=embedding.tolist(),
                        payload={'text': text,
                                 'ticker': ticker,
                                 'year': year,
                                 'doc_type': doc_type,
                                 },
                        )
            for i, (embedding, text, (ticker, year, doc_type)) in enumerate(zip(embeddings, texts, meta))
        ]

        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            batch_num = i // batch_size + 1
            result = self.client.upsert(collection_name=self.collection,
                                       points=batch,
                                       wait=True)
            logger.info(f"Upsert batch #{batch_num} {result.status}, operation_id={result.operation_id}, points={len(batch)}")

    def build_filters(filters: QueryFilters) -> dict:
        """expects dict of {field: {operator: filter_value}}"""
        conditions = []
        for query, filter in query_filters.items():
            if query.lower() not in ['ticker', 'year', 'doc_type']:
                continue
            for operator, value in filter:
                match_filter = None
                if operator == "$eq":
                    match_filter=MatchValue(value=value),
                elif operator == "$in":
                    match_filter=MatchAny(any=value),
                elif operator in ["$lt", "$gt", "$lte", "$gte"]:
                    match_filter=Range(**{operator.replace("$", ""): filter})

                if match_filter:
                    conditions.append(
                        FieldCondition(
                            key=query.strip().lower(),
                            match=match_filter
                        ))
        return conditions

