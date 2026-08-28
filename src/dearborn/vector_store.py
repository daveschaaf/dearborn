import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


FINRANK_PATH = Path(__file__).parent.parent.parent / "data/FinRank"
sys.path.insert(0, str(FINRANK_PATH))
from run_baselines import load_records, build_corpus

DATA_PATH = FINRANK_PATH / "FinRank.jsonl"
COLLECTION = "finrank_mpnet"
MODEL = "sentence-transformers/all-mpnet-base-v2"
