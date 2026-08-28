import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from run_baselines import load_records, build_corpus

FINRANK_PATH = Path(sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data/FinRank")))
DATA_PATH = FINRANK_PATH / "FinRank.jsonl"
COLLECTION = "finrank_mpnet"
MODEL = "sentence-transformers/all-mpnet-base-v2"
