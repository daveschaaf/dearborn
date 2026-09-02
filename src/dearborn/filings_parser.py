from edgar import set_identity
from edgar import Company as EdgarCompany
from .company_data import Company, CompanyFiling, FilingChunk, sqlite_connection
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Literal
from dearborn.logging_config import setup_logging
import sec2md
import re
import pandas as pd

logger = logging.getLogger(__name__)

PARSER_VERSION = "sec2md-0.1.23-item-fallback-chunk384-overlap96-v2"

TICKERS = [
    "BMY", "GILD", "REGN", "VTRS", "BIIB", "ZTS",
    "PSX", "VLO", "PBF", "DINO", "OXY", "FANG", "EXE", "APA",
    "PCAR", "LEA", "GT", "BWA", "ALV",
]

RETRIEVAL_ITEMS_10K = frozenset({
    sec2md.Item10K.BUSINESS,
    sec2md.Item10K.RISK_FACTORS,
    sec2md.Item10K.CYBERSECURITY,
    sec2md.Item10K.LEGAL_PROCEEDINGS,
    sec2md.Item10K.MD_AND_A,
    sec2md.Item10K.MARKET_RISK,
    sec2md.Item10K.FINANCIAL_STATEMENTS,
    sec2md.Item10K.CONTROLS_AND_PROCEDURES,
})

RETRIEVAL_ITEMS_10Q = frozenset({
    sec2md.Item10Q.FINANCIAL_STATEMENTS_P1,
    sec2md.Item10Q.MD_AND_A_P1,
    sec2md.Item10Q.MARKET_RISK_P1,
    sec2md.Item10Q.CONTROLS_AND_PROCEDURES_P1,
    sec2md.Item10Q.LEGAL_PROCEEDINGS_P2,
    sec2md.Item10Q.RISK_FACTORS_P2,
})

DATE_FORMAT = "%Y-%m-%d"

PART_HEADING = re.compile(
    r"^\*\*(?P<part>PART\s+(?:I|II))(?:\s*[—-]\s*[^\n*]+)?\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ITEM_HEADING = re.compile(
    r"^\*\*ITEM\s+(?P<item>\d+[A-Z]?)\.\s*(?P<title>.*?)\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FALLBACK_HEADING = re.compile(
    r"^\*\*(?P<part>PART\s+(?:I|II))(?:\s*[—-]\s*[^\n*]+)?\*\*\s*$"
    r"|^\*\*ITEM\s+(?P<item>\d+[A-Z]?)\.\s*(?P<title>.*?)\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

@dataclass(frozen=True)
class FilingParser:
    """Parse one SEC filing into provenance-preserving retrieval chunks."""

    chunk_size: int = 384
    chunk_overlap: int = 96
    max_table_tokens: int = 2048
    parser_version: str = PARSER_VERSION

    @staticmethod
    def _date_text(value):
        if value is None or pd.isna(value):
            return None
        if isinstance(value, str):
            return value
        return value.strftime(DATE_FORMAT)

    @staticmethod
    def select_reporting_period(form: str, periods):
        """Choose the filing's reported fiscal period, not a shorter fact context."""
        valid_periods = periods.dropna(
            subset=["fiscal_period", "fiscal_year"]
        ).copy()

        if form == "10-K":
            candidates = valid_periods.loc[
                valid_periods["fiscal_period"].eq("FY")
            ]
        else:
            candidates = valid_periods.loc[
                valid_periods["fiscal_period"].astype(str).str.fullmatch(r"Q[1-4]")
            ]

        if candidates.empty:
            raise ValueError(f"No reporting fiscal period found for {form}")

        return candidates.sort_values("period_start").iloc[0]

    def parse_filing_metadata(self, filing) -> CompanyFiling:
        xbrl = filing.xbrl()
        if xbrl is None:
            raise ValueError(f"{filing.accession_number} has no XBRL data")

        facts = xbrl.facts.to_dataframe()
        report_date = self._date_text(filing.report_date)

        periods = facts.loc[
            facts["period_type"].eq("duration")
            & facts["period_end"].map(self._date_text).eq(report_date)
            & facts["is_dimensioned"].eq(False),
            ["period_start", "period_end", "fiscal_period", "fiscal_year"],
        ].drop_duplicates()

        if periods.empty:
            raise ValueError(
                f"No duration period ending {report_date} "
                f"for {filing.accession_number}"
            )

        period = self.select_reporting_period(filing.form, periods)
        filing_date = self._date_text(filing.filing_date)

        return CompanyFiling(
            cik=str(filing.cik),
            accession_number=filing.accession_number,
            form=filing.form,
            filing_date=filing_date,
            filing_year=int(filing_date[:4]),
            source_url=filing.filing_url,
            fiscal_year=f"FY{int(period['fiscal_year'])}",
            fiscal_period=period["fiscal_period"],
            fiscal_period_start=self._date_text(period["period_start"]),
            fiscal_period_end=self._date_text(period["period_end"]),
            parser_version=self.parser_version,
            parse_complete=False,
        )

    def item_code(self, filing, item) -> str | None:
        """Normalize sec2md's rendered Item label to its enum value."""
        if item.item is None:
            return None
        code = item.item.removeprefix("ITEM ").strip()
        if filing.form == "10-Q":
            part_suffix = {"PART I": "P1", "PART II": "P2"}.get(item.part)
            if part_suffix:
                return f"{code}.{part_suffix}"
        return code

    def _fallback_10q_items(self, pages):
        """Split normalized 10-Q pages when sec2md's extractor finds no Items."""
        items = []
        current = None
        current_part = None

        for page in pages:
            content = page.content
            matches = list(FALLBACK_HEADING.finditer(content))

            if not matches:
                if current is not None and content.strip():
                    current["pages"].append(page)
                continue

            previous_end = 0
            for match in matches:
                preceding_text = content[previous_end:match.start()].strip()
                if current is not None and preceding_text:
                    current["pages"].append(
                        page.model_copy(update={"content": preceding_text})
                    )

                part = match.group("part")
                if part is not None:
                    current_part = part.upper()
                else:
                    current = {
                        "part": current_part,
                        "item": f"ITEM {match.group('item').upper()}",
                    "item_title": match.group("title").strip(" *"),
                        "pages": [],
                    }
                    items.append(current)

                previous_end = match.end()

            remaining_text = content[previous_end:].strip()
            if current is not None and remaining_text:
                current["pages"].append(
                    page.model_copy(update={"content": remaining_text})
                )

        return [sec2md.Section(**item) for item in items]

    def parse_filing_items(self, filing, pages):
        all_items = sec2md.extract_sections(
            pages,
            filing_type=filing.form,
        )

        if filing.form == "10-Q" and not all_items:
            all_items = self._fallback_10q_items(pages)

        normalized_items = []
        for item in all_items:
            if filing.form == "10-Q" and item.part == "PART I" and item.item is None:
                item = item.model_copy(
                    update={
                        "item": "ITEM 1",
                        "item_title": "FINANCIAL STATEMENTS",
                    }
                )
            if item.item is not None:
                normalized_items.append(item)

        retrieval_items = (
            RETRIEVAL_ITEMS_10K
            if filing.form == "10-K"
            else RETRIEVAL_ITEMS_10Q
        )

        return [
            item
            for item in normalized_items
            if self.item_code(filing, item) in retrieval_items
        ]

    def chunk_filing_item(self, filing, item):
        chunks = sec2md.chunk_section(
            item,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_table_tokens=self.max_table_tokens,
        )

        part = item.part or filing.form
        return [
            FilingChunk(
                chunk_id=(
                    f"{filing.accession_number}:"
                    f"{part}:{item.item}:{chunk_index}"
                ),
                accession_number=filing.accession_number,
                part=part,
                item=item.item,
                item_title=item.item_title,
                chunk_index=chunk_index,
                text=chunk.content,
                parser_version=self.parser_version,
            )
            for chunk_index, chunk in enumerate(chunks)
        ]

    def parse(
        self,
        edgar_filing,
        raw_html: str | None = None,
    ) -> tuple[CompanyFiling, list[FilingChunk]]:
        filing = self.parse_filing_metadata(edgar_filing)
        raw_html = raw_html if raw_html is not None else edgar_filing.html()
        pages = sec2md.Parser(raw_html).get_pages()
        items = self.parse_filing_items(edgar_filing, pages)
        chunks = [
            chunk
            for item in items
            for chunk in self.chunk_filing_item(edgar_filing, item)
        ]
        logger.info(
            "Parsed filing accession=%s form=%s items=%d chunks=%d",
            filing.accession_number,
            filing.form,
            len(items),
            len(chunks),
        )
        return filing, chunks


@dataclass(frozen=True)
class FilingSaveResult:
    status: Literal["dry_run", "saved", "skipped"]
    accession_number: str
    chunk_count: int
    raw_path: Path | None


def completed_chunk_count(connection, accession_number: str, parser_version: str) -> int | None:
    row = connection.execute(
        """
        SELECT COUNT(*)
        FROM filing
        JOIN filing_chunk USING (accession_number)
        WHERE filing.accession_number = ?
          AND filing.parser_version = ?
          AND filing.parse_complete = 1
        """,
        (accession_number, parser_version),
    ).fetchone()
    return row[0] if row and row[0] else None


def save_filing(
    edgar_filing,
    *,
    save: bool = False,
    connection=None,
    raw_dir: Path = Path("data/raw"),
    parser: FilingParser | None = None,
) -> FilingSaveResult:
    """Parse a filing and, when requested, save its raw HTML and normalized records."""
    if save and connection is None:
        raise ValueError("connection is required when save=True")

    parser = parser or FilingParser()
    accession_number = edgar_filing.accession_number
    raw_path = Path(raw_dir) / f"{accession_number}.html"

    if save:
        chunk_count = completed_chunk_count(
            connection,
            accession_number,
            parser.parser_version,
        )
        if chunk_count is not None:
            logger.info(
                "Skipping completed filing accession=%s chunks=%d",
                accession_number,
                chunk_count,
            )
            return FilingSaveResult(
                status="skipped",
                accession_number=accession_number,
                chunk_count=chunk_count,
                raw_path=raw_path if raw_path.exists() else None,
            )

    if raw_path.exists():
        raw_html = raw_path.read_text(encoding="utf-8")
        logger.info("Reusing cached raw filing accession=%s", accession_number)
    else:
        raw_html = edgar_filing.html()
        if save:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw_html, encoding="utf-8")
            logger.info("Saved raw filing accession=%s path=%s", accession_number, raw_path)

    filing, chunks = parser.parse(edgar_filing, raw_html=raw_html)
    filing.parser_version = parser.parser_version
    filing.parse_complete = False

    if not save:
        logger.info("Dry run completed accession=%s", filing.accession_number)
        return FilingSaveResult(
            status="dry_run",
            accession_number=filing.accession_number,
            chunk_count=len(chunks),
            raw_path=raw_path if raw_path.exists() else None,
        )

    with connection:
        connection.execute(
            "DELETE FROM filing_chunk WHERE accession_number = ?",
            (filing.accession_number,),
        )
        connection.execute(
            "DELETE FROM filing WHERE accession_number = ?",
            (filing.accession_number,),
        )
        filing.save(connection, commit=False)
        for chunk in chunks:
            chunk.save(connection, commit=False)
        connection.execute(
            "UPDATE filing SET parse_complete = 1 WHERE accession_number = ?",
            (filing.accession_number,),
        )

    logger.info(
        "Saved filing accession=%s chunks=%d raw_path=%s",
        filing.accession_number,
        len(chunks),
        raw_path,
    )
    return FilingSaveResult(
        status="saved",
        accession_number=filing.accession_number,
        chunk_count=len(chunks),
        raw_path=raw_path,
    )

def run(save: bool = False) -> None:
    set_identity("davidschaaf@berkeley.edu")

    connection = sqlite_connection()

    for ticker in TICKERS:
        logger.info("Processing ticker=%s save=%s", ticker, save)
        edgar_co = EdgarCompany(ticker)
        company = Company(ticker=ticker, name=edgar_co.name,
                          cik=edgar_co.cik, sic=edgar_co.sic, industry=edgar_co.industry)
        if save:
            company.save(connection)

        ten_k = edgar_co.get_filings(form="10-K").latest()
        save_filing(ten_k, save=save, connection=connection)
        ten_q = edgar_co.get_filings(form="10-Q").latest()
        save_filing(ten_q, save=save, connection=connection)

if __name__ == "__main__":
    setup_logging()
    run(save=True)
