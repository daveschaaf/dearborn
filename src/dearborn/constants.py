from pathlib import Path

QDRANT_STORAGE = 'qdrant_storage'
QDRANT_COLLECTION_MP = "finrank_mpnet"

def _find_project_root() -> Path:
    for parent in [Path(__file__)] + list(Path(__file__).parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("pyproject.toml not found")

PROJECT_ROOT = _find_project_root()
CONTEXT_DIR = PROJECT_ROOT / "src" / "dearborn" / "context"
FINRANK_DIR = PROJECT_ROOT / "data" / "FinRank"
