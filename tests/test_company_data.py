import sqlite3

import pytest

from dearborn.company_data import (
    Company,
    CompanyFiling,
    FilingChunk,
    create_table_company,
    create_table_filing,
    create_table_filing_chunk,
    migrate_schema,
)


@pytest.fixture
def connection():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(create_table_company)
    conn.execute(create_table_filing)
    conn.execute(create_table_filing_chunk)
    yield conn
    conn.close()


def company() -> Company:
    return Company(
        cik="0012345678",
        name="Example Corporation",
        ticker="EXM",
        sic=2834,
        industry="Pharmaceuticals",
    )


def filing() -> CompanyFiling:
    return CompanyFiling(
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


def test_data_models_save_a_company_filing_and_chunk(connection):
    saved_company = company()
    saved_filing = filing()
    chunk = FilingChunk(
        chunk_id="0001234567-25-000001:PART II:ITEM 7:0",
        accession_number=saved_filing.accession_number,
        part="PART II",
        item="ITEM 7",
        item_title="MANAGEMENT'S DISCUSSION AND ANALYSIS",
        chunk_index=0,
        text="Example MD&A evidence.",
        parser_version="test-v1",
    )

    assert saved_company.save(connection) == saved_company.cik
    assert saved_filing.save(connection) == saved_filing.accession_number
    assert chunk.save(connection) == chunk.chunk_id

    assert connection.execute(
        "SELECT ticker, sic, industry FROM company WHERE cik = ?",
        (saved_company.cik,),
    ).fetchone() == ("EXM", 2834, "Pharmaceuticals")
    assert connection.execute(
        "SELECT fiscal_year, fiscal_period FROM filing WHERE accession_number = ?",
        (saved_filing.accession_number,),
    ).fetchone() == ("FY2024", "FY")
    assert connection.execute(
        "SELECT part, item, item_title, parser_version "
        "FROM filing_chunk WHERE chunk_id = ?",
        (chunk.chunk_id,),
    ).fetchone() == (
        "PART II",
        "ITEM 7",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "test-v1",
    )


def test_filing_chunk_requires_an_existing_filing(connection):
    orphan_chunk = FilingChunk(
        chunk_id="missing:PART I:ITEM 1:0",
        accession_number="missing",
        part="PART I",
        item="ITEM 1",
        item_title="BUSINESS",
        chunk_index=0,
        text="Orphan evidence.",
        parser_version="test-v1",
    )

    with pytest.raises(sqlite3.IntegrityError):
        orphan_chunk.save(connection)


def test_migrate_schema_adds_filing_completion_columns():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE filing (
            accession_number TEXT PRIMARY KEY,
            source_url TEXT NOT NULL
        ) STRICT
        """
    )

    migrate_schema(connection)

    columns = {
        row[1]: row
        for row in connection.execute("PRAGMA table_info(filing)")
    }
    assert "parser_version" in columns
    assert "parse_complete" in columns
    assert columns["parser_version"][4] == "''"
    assert columns["parse_complete"][4] == "0"
