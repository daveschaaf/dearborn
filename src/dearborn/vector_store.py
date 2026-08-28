from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging
logger = logging.getLogger(__name__)

COLLECTION = "finrank_mpnet"
MODEL = "sentence-transformers/all-mpnet-base-v2"

class VectorStore():
    def __init__(self, storage: str):
        logger.info(f"Initializing VectorStore with storage={storage}")
        self.storage = storage

    def embed(self, text):
        pass

    def init_vector_store(self, path):
        self.client = QdrantClient(host="localhost", port=6333)
        self.client.create_collection(
          collection_name=COLLECTION,
          vectors_config=VectorParams(size=768, distance=Distance.DOT)
        )
        base_embeddings = HuggingFaceEmbeddings(model_name=MODEL)
        self.vector_store = QdrantVectorStore(
          client=self.client,
          embedding=base_embeddings,
          collection_name=COLLECTION,
          distance=Distance.DOT
      )
