from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, UpdateResult, SearchParams, Filter, FieldCondition, MatchValue, MatchAny, Range
from .query_filters import QueryFilters 
import logging
logger = logging.getLogger(__name__)

MODEL = "sentence-transformers/all-mpnet-base-v2"

class VectorStore():
    def __init__(self, url: str, collection: str):
        logger.info(f"VectorStore for collection={collection} initialized")
        self.collection = collection
        self.client = QdrantClient(url=url)
        logger.info(f"Loading model {MODEL}")
        self.encoder = SentenceTransformer(MODEL)
        self.create_collection()
        print(QueryFilters.model_json_schema())

    def delete_collection(self):
        self.client.delete_collection(self.collection)
        logger.info(f"Collection={self.collection} deleted")

    def create_collection(self):
        if self.client.collection_exists(self.collection):
            logger.info(f"Collection={self.collection} already exists")
            return
        self.client.create_collection(
          collection_name=self.collection,
          vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        logger.info(f"Collection={self.collection} created.")

    def retrieve(self, query: str, filters: QueryFilters, top_k: int = 10) -> list[ScoredPoint]:
        query_filter = self.build_filters(filters)
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            search_params=SearchParams(exact=True)
        ).points

        return results

    def upsert(self, texts: list[str], meta: list[tuple]) -> None:
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

    def build_filters(self, query_filters: QueryFilters) -> Filter | None:
        filters = {
            "ticker": query_filters.ticker,
            "year": query_filters.year,
            "doc_type": query_filters.doc_type
        }
        conditions = []
        for field, ops in list(filters.items()):
            for operator, value in ops.items():
                if value is None:
                    continue
                if operator == "$eq":
                    field_condition = FieldCondition(key=field, match=MatchValue(value=value))
                elif operator == "$in":
                    field_condition = FieldCondition(key=field, match=MatchAny(any=value))
                elif operator in ["$lt", "$gt", "$lte", "$gte"]:
                    field_condition = FieldCondition(key=field, range=Range(**{operator.replace("$", ""): value}))
                else:
                    continue
                conditions.append(field_condition)
        return Filter(must=conditions) if conditions else None

