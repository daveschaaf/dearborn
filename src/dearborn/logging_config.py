import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

def setup_logging() -> None:

    logger = logging.getLogger("dearborn")
    if logger.handlers:
        return

    logger.setLevel(logging.DEBUG)

    format = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(LOG_DIR / 'app.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(format)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(format)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

