from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, UpdateResult
import logging
logger = logging.getLogger(__name__)

MODEL = "sentence-transformers/all-mpnet-base-v2"

class VectorStore():
    def __init__(self, url: str, collection: str):
        logger.info(f"VectorStore for collection={collection} initialized")
        self.collection = collection
        self.client = QdrantClient(url=url)
        self.encoder = SentenceTransformer(MODEL)
        self.create_collection()

    def create_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
              collection_name=self.collection,
              vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logger.info(f"Collection={self.collection} created.")
        else:
            logger.info(f"Collection={self.collection} already exists")

    def retrieve(self, query: str, top_k: int = 10) -> list[ScoredPoint]:
        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        ).points

        return results

    def upsert(self, texts: list[str]) -> str:
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
                        payload={'text': text},
                        )
            for i, (embedding, text) in enumerate(zip(embeddings, texts))
        ]

        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            batch_num = i // batch_size + 1
            result = self.client.upsert(collection_name=self.collection,
                                       points=batch,
                                       wait=True)
            logger.info(f"Upsert batch #{batch_num} {result.status}, operation_id={result.operation_id}, points={len(batch)}")
