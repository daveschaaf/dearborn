import subprocess
from pathlib import Path
from .constants import QDRANT_STORAGE

def qdrant_up() -> None:
    storage = Path.cwd() / QDRANT_STORAGE
    subprocess.run([
        "docker", "run",
        "-p", "6333:6333", "-p", "6334:6333",
        "-v", f"{storage}:/qdrant/storage",
        "qdrant/qdrant"
    ])
