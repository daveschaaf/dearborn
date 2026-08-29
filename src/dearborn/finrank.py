import sys
import logging
from pathlib import Path
logger = logging.getLogger(__name__)
logger.info("loading FinRank")
FINRANK_PATH: Path = Path(__file__).parent.parent.parent / "data/FinRank"
sys.path.insert(0, str(FINRANK_PATH/'baselines'))
from run_baselines import load_records, build_corpus, per_record_metrics, aggregate, tokenize
logger.info("loaded FinRank methods")
DATA_PATH: Path = FINRANK_PATH / "FinRank.jsonl"


aggregate = aggregate
per_record_metrics = per_record_metrics
tokenize = tokenize

def gold_records(records: list[dict]) -> list[set]:
    golds = []
    for rec in records:
        gold_passages = set(INDEX[p['text']] for p in rec['passages'] if rec['passages']  )
        golds.append(gold_passages)
    return golds

RECORDS: list[dict] = load_records(DATA_PATH)
TEXT, METADATA, INDEX = build_corpus(RECORDS)
QUESTIONS: list[dict] = [rec['question'] for rec in RECORDS]
GOLDS: list[set] = gold_records(RECORDS)
