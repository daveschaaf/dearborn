from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re

import requests
from requests import RequestException

try:
    from .logging_config import setup_logging
except ImportError:
    from logging_config import setup_logging

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "raw" / "transcripts"

QUESTION_AND_ANSWER_SESSION = re.compile(
    r"\bquestion(?:\s|-|&|and)+answer(?:\s+session)?\b|\bq\s*&\s*a\b",
    re.IGNORECASE,
)
QUESTION_ANNOUNCEMENT = re.compile(
    r"""
    \b(?:the\s+|our\s+|your\s+)?(?:first|next)\s+question\s+
    (?:(?:will|shall)\s+)?come(?:s)?\s+from\s+
    (?P<analyst>.+?)(?:\s+(?:with|of)\s+[^.]+)?(?:\.|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class QuestionAnswerPair:
    analyst: str
    question: str
    answer: str
    answer_speakers: tuple[str, ...]
    question_segment_indexes: tuple[int, ...]
    answer_segment_indexes: tuple[int, ...]


@dataclass(frozen=True)
class ParseReview:
    code: str
    segment_index: int | None
    detail: str


@dataclass(frozen=True)
class TranscriptParseResult:
    pairs: tuple[QuestionAnswerPair, ...]
    review_reasons: tuple[ParseReview, ...]


def _normalized_speaker(value: str) -> str:
    return " ".join(value.lower().split())


def _speaker_matches_analyst(speaker: str, analyst: str) -> bool:
    return _normalized_speaker(speaker) == _normalized_speaker(analyst)


def parse_question_answer_pairs(
    structured_content: list[dict[str, str]],
) -> TranscriptParseResult:
    """Deterministically extract Q&A turns from one transcript's speaker segments."""
    pairs: list[QuestionAnswerPair] = []
    reviews: list[ParseReview] = []
    in_question_and_answer = False
    moderator: str | None = None
    current: dict[str, object] | None = None

    def save_current(next_segment_index: int | None) -> None:
        if current is None:
            return

        analyst = current["analyst"]
        question_parts = current["question_parts"]
        answer_parts = current["answer_parts"]
        if not question_parts:
            reviews.append(ParseReview("missing_question", next_segment_index, analyst))
            return
        if not answer_parts:
            reviews.append(ParseReview("missing_answer", next_segment_index, analyst))
            return

        pairs.append(
            QuestionAnswerPair(
                analyst=analyst,
                question=" ".join(question_parts),
                answer=" ".join(answer_parts),
                answer_speakers=tuple(current["answer_speakers"]),
                question_segment_indexes=tuple(current["question_indexes"]),
                answer_segment_indexes=tuple(current["answer_indexes"]),
            )
        )

    for index, segment in enumerate(structured_content):
        speaker = segment["speaker"].strip()
        text = segment["text"].strip()
        if not text:
            continue

        if QUESTION_AND_ANSWER_SESSION.search(text) and not in_question_and_answer:
            in_question_and_answer = True
            moderator = speaker

        announcement = QUESTION_ANNOUNCEMENT.search(text)
        if announcement:
            if not in_question_and_answer:
                in_question_and_answer = True
                moderator = speaker
                reviews.append(
                    ParseReview(
                        "q_and_a_boundary_inferred_from_announcement",
                        index,
                        "No explicit Q&A session marker.",
                    )
                )
            save_current(index)
            current = {
                "analyst": announcement.group("analyst").strip(),
                "question_parts": [],
                "answer_parts": [],
                "answer_speakers": [],
                "question_indexes": [],
                "answer_indexes": [],
            }
            continue

        if not in_question_and_answer or current is None:
            continue
        if speaker == "Operator" or speaker == moderator:
            continue

        if _speaker_matches_analyst(speaker, current["analyst"]):
            current["question_parts"].append(text)
            current["question_indexes"].append(index)
            continue

        # The named analyst is the only non-management speaker in this turn.
        current["answer_parts"].append(text)
        current["answer_indexes"].append(index)
        if speaker not in current["answer_speakers"]:
            current["answer_speakers"].append(speaker)

    if not in_question_and_answer:
        reviews.append(ParseReview("no_q_and_a_section", None, "No Q&A session was detected."))
    save_current(None)
    return TranscriptParseResult(tuple(pairs), tuple(reviews))

class TranscriptClient:
    API_URL = "https://datasets-server.huggingface.co/filter"

    def __init__(self, raw_directory: Path = DEFAULT_RAW_TRANSCRIPT_DIR) -> None:
        self.raw_directory = Path(raw_directory)

    def raw_path(self, ticker: str, year: int, quarter: str) -> Path:
        hf_quarter = int(quarter.removeprefix("Q"))
        return self.raw_directory / f"{ticker.upper()}_{year}_Q{hf_quarter}.json"

    def load_transcript(
        self,
        ticker: str,
        year: int,
        quarter: str,
        *,
        refresh: bool = False,
    ) -> dict:
        raw_path = self.raw_path(ticker, year, quarter)
        if raw_path.exists() and not refresh:
            logger.info("Reusing raw transcript: %s", raw_path)
            return json.loads(raw_path.read_text(encoding="utf-8"))

        hf_quarter = int(quarter.removeprefix("Q"))
        logger.info("Loading transcript: %s %s Q%s", ticker.upper(), year, hf_quarter)

        try:
            response = requests.get(
                self.API_URL,
                params={
                    "dataset": "kurry/sp500_earnings_transcripts",
                    "config": "default",
                    "split": "train",
                    "where": (
                        f'"symbol" = \'{ticker.upper()}\' '
                        f'AND "year" = {year} '
                        f'AND "quarter" = {hf_quarter}'
                    ),
                    "offset": 0,
                    "length": 1,
                },
                timeout=30,
            )
            response.raise_for_status()
        except RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            logger.error(
                "Transcript request failed for %s %s %s (status=%s): %s",
                ticker.upper(), year, quarter, status, exc,
            )
            raise
        rows = response.json()["rows"]

        if len(rows) != 1:
            raise LookupError(
                f"Expected one transcript for {ticker} {year} {quarter}; "
                f"found {len(rows)}."
            )

        transcript = rows[0]["row"]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = raw_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary_path.replace(raw_path)
        logger.info("Saved raw transcript: %s", raw_path)
        return transcript

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("dearborn.transcript_parser")
    logger.info("Running transcript_parser.py")
    transcript_client = TranscriptClient()
    transcript = transcript_client.load_transcript("BMY", 2025, "Q1")
    result = parse_question_answer_pairs(transcript["structured_content"])
    logger.info(
        "Parsed %d question-answer pairs; review flags=%d",
        len(result.pairs),
        len(result.review_reasons),
    )
