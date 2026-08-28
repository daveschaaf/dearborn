import logging
from .logging_config import setup_logging
from .vector_store import build



def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("hello world")

if __name__ == "__main__":
    main()
