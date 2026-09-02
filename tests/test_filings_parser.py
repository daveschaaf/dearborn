from types import SimpleNamespace
import logging
import sqlite3

import pandas as pd
import sec2md
import pytest

from dearborn.company_data import (
    Company,
    CompanyFiling,
    FilingChunk,
    create_table_company,
    create_table_filing,
    create_table_filing_chunk,
)
from dearborn.filings_parser import FilingParser, PARSER_VERSION, save_filing


def filing_stub(form: str) -> SimpleNamespace:
    return SimpleNamespace(
        form=form,
        accession_number="0000000000-26-000001",
    )


def page(number: int, content: str) -> sec2md.Page:
    return sec2md.Page(number=number, content=content)


@pytest.fixture
def connection():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(create_table_company)
    conn.execute(create_table_filing)
    conn.execute(create_table_filing_chunk)
    Company(
        cik="0012345678",
        name="Example Corporation",
        ticker="EXM",
        sic=2834,
        industry="Pharmaceuticals",
    ).save(conn)
    yield conn
    conn.close()


class EdgarFilingStub:
    form = "10-K"
    accession_number = "0001234567-25-000001"

    def __init__(self, raw_html="<html><body>Original SEC filing</body></html>"):
        self.raw_html = raw_html
        self.html_calls = 0

    def html(self):
        self.html_calls += 1
        return self.raw_html


class ParserStub:
    parser_version = "test-v1"

    def __init__(self):
        self.raw_html = None
        self.parse_calls = 0
        self.filing = CompanyFiling(
            cik="0012345678",
            accession_number="0001234567-25-000001",
            form="10-K",
            filing_date="2025-02-15",
            filing_year=2025,
            source_url="https://www.sec.gov/example",
            fiscal_year="FY2024",
            fiscal_period="FY",
            fiscal_period_start="2024-01-01",
            fiscal_period_end="2024-12-31",
        )
        self.chunks = [
            FilingChunk(
                chunk_id="0001234567-25-000001:PART II:ITEM 7:0",
                accession_number=self.filing.accession_number,
                part="PART II",
                item="ITEM 7",
                item_title="MD&A",
                chunk_index=0,
                text="Evidence.",
                parser_version="test-v1",
            )
        ]

    def parse(self, edgar_filing, raw_html):
        self.parse_calls += 1
        self.raw_html = raw_html
        return self.filing, self.chunks


def test_item_code_normalizes_10k_and_10q_items():
    parser = FilingParser()

    ten_k_item = SimpleNamespace(
        part="PART II",
        item="ITEM 7",
        item_title="MD&A",
    )
    ten_q_item = SimpleNamespace(
        part="PART II",
        item="ITEM 1A",
        item_title="RISK FACTORS",
    )

    assert parser.item_code(filing_stub("10-K"), ten_k_item) == "7"
    assert parser.item_code(filing_stub("10-Q"), ten_q_item) == "1A.P2"


def test_select_reporting_period_uses_fy_for_a_10k_over_shorter_facts():
    periods = pd.DataFrame(
        [
            {
                "period_start": "2025-01-01",
                "period_end": "2025-12-31",
                "fiscal_period": "FY",
                "fiscal_year": 2025,
            },
            {
                "period_start": "2025-12-19",
                "period_end": "2025-12-31",
                "fiscal_period": None,
                "fiscal_year": 2025,
            },
            {
                "period_start": "2025-10-01",
                "period_end": "2025-12-31",
                "fiscal_period": "Q4",
                "fiscal_year": 2025,
            },
        ]
    )

    period = FilingParser().select_reporting_period("10-K", periods)

    assert period["fiscal_period"] == "FY"
    assert period["period_start"] == "2025-01-01"


def test_select_reporting_period_uses_the_quarter_for_a_10q():
    periods = pd.DataFrame(
        [
            {
                "period_start": "2025-04-01",
                "period_end": "2025-06-30",
                "fiscal_period": "Q2",
                "fiscal_year": 2025,
            },
            {
                "period_start": "2025-06-20",
                "period_end": "2025-06-30",
                "fiscal_period": None,
                "fiscal_year": 2025,
            },
        ]
    )

    period = FilingParser().select_reporting_period("10-Q", periods)

    assert period["fiscal_period"] == "Q2"


def test_parse_filing_items_uses_10q_fallback_and_preserves_boundaries(monkeypatch):
    parser = FilingParser()
    filing = filing_stub("10-Q")
    pages = [
        page(
            3,
            """**PART I—FINANCIAL INFORMATION**

**Item 1. FINANCIAL STATEMENTS**

Item 1 evidence.

**Item 2. MANAGEMENT'S DISCUSSION AND ANALYSIS**

Item 2 evidence.
""",
        ),
        page(
            4,
            """**PART II—OTHER INFORMATION**

**Item 1. LEGAL PROCEEDINGS**

Legal evidence.

**Item 1A. RISK FACTORS**

Risk evidence.
""",
        ),
    ]

    monkeypatch.setattr(sec2md, "extract_sections", lambda *args, **kwargs: [])

    items = parser.parse_filing_items(filing, pages)

    assert [(item.part, item.item) for item in items] == [
        ("PART I", "ITEM 1"),
        ("PART I", "ITEM 2"),
        ("PART II", "ITEM 1"),
        ("PART II", "ITEM 1A"),
    ]
    assert "Item 1 evidence." in items[0].pages[0].content
    assert "Item 2 evidence." not in items[0].pages[0].content
    assert "Risk evidence." in items[3].pages[0].content


def test_parse_filing_items_normalizes_unlabeled_10q_part_i_item_one(monkeypatch):
    parser = FilingParser()
    filing = filing_stub("10-Q")
    item_one_page = page(6, "Financial statement evidence.")
    item_two_page = page(51, "MD&A evidence.")
    unlabeled_item_one = sec2md.Section(
        part="PART I",
        item=None,
        item_title=None,
        pages=[item_one_page],
    )
    item_two = sec2md.Section(
        part="PART I",
        item="ITEM 2",
        item_title="MD&A",
        pages=[item_two_page],
    )
    monkeypatch.setattr(
        sec2md,
        "extract_sections",
        lambda *args, **kwargs: [unlabeled_item_one, item_two],
    )

    items = parser.parse_filing_items(filing, [item_one_page, item_two_page])

    assert [(item.item, item.item_title) for item in items] == [
        ("ITEM 1", "FINANCIAL STATEMENTS"),
        ("ITEM 2", "MD&A"),
    ]


def test_chunk_filing_item_preserves_parent_item_metadata(monkeypatch):
    parser = FilingParser()
    filing = filing_stub("10-Q")
    item = SimpleNamespace(
        part="PART II",
        item="ITEM 1A",
        item_title="RISK FACTORS",
    )
    monkeypatch.setattr(
        sec2md,
        "chunk_section",
        lambda *args, **kwargs: [SimpleNamespace(content="Risk evidence.")],
    )

    chunks = parser.chunk_filing_item(filing, item)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "0000000000-26-000001:PART II:ITEM 1A:0"
    assert chunks[0].part == "PART II"
    assert chunks[0].item == "ITEM 1A"
    assert chunks[0].item_title == "RISK FACTORS"
    assert chunks[0].parser_version == PARSER_VERSION


def test_save_filing_persists_raw_html_and_normalized_records(
    connection,
    tmp_path,
    caplog,
):
    parser = ParserStub()
    caplog.set_level(logging.INFO, logger="dearborn.filings_parser")

    result = save_filing(
        EdgarFilingStub(),
        save=True,
        connection=connection,
        raw_dir=tmp_path,
        parser=parser,
    )

    assert result.status == "saved"
    assert parser.raw_html == "<html><body>Original SEC filing</body></html>"
    assert (tmp_path / f"{result.accession_number}.html").read_text() == parser.raw_html
    assert connection.execute("SELECT COUNT(*) FROM filing").fetchone() == (1,)
    assert connection.execute("SELECT COUNT(*) FROM filing_chunk").fetchone() == (1,)
    assert result.chunk_count == 1
    assert "Saved filing accession=0001234567-25-000001 chunks=1" in caplog.text


def test_save_filing_dry_run_does_not_write_raw_html_or_database(tmp_path, caplog):
    parser = ParserStub()
    caplog.set_level(logging.INFO, logger="dearborn.filings_parser")

    result = save_filing(
        EdgarFilingStub(),
        raw_dir=tmp_path,
        parser=parser,
    )

    assert result.status == "dry_run"
    assert result.chunk_count == 1
    assert not list(tmp_path.iterdir())
    assert "Dry run completed accession=0001234567-25-000001" in caplog.text


def test_save_filing_reuses_cached_raw_html_when_database_is_incomplete(
    connection,
    tmp_path,
):
    cached_html = "<html><body>Cached SEC filing</body></html>"
    raw_path = tmp_path / "0001234567-25-000001.html"
    raw_path.write_text(cached_html)
    edgar_filing = EdgarFilingStub(raw_html="should not be downloaded")
    parser = ParserStub()

    result = save_filing(
        edgar_filing,
        save=True,
        connection=connection,
        raw_dir=tmp_path,
        parser=parser,
    )

    assert result.status == "saved"
    assert edgar_filing.html_calls == 0
    assert parser.raw_html == cached_html


def test_save_filing_skips_a_completed_current_version(connection, tmp_path):
    first_parser = ParserStub()
    first_result = save_filing(
        EdgarFilingStub(),
        save=True,
        connection=connection,
        raw_dir=tmp_path,
        parser=first_parser,
    )
    second_parser = ParserStub()
    edgar_filing = EdgarFilingStub(raw_html="should not be downloaded")

    second_result = save_filing(
        edgar_filing,
        save=True,
        connection=connection,
        raw_dir=tmp_path,
        parser=second_parser,
    )

    assert first_result.status == "saved"
    assert second_result.status == "skipped"
    assert second_result.chunk_count == 1
    assert edgar_filing.html_calls == 0
    assert second_parser.parse_calls == 0
