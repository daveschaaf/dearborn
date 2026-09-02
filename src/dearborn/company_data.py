from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
import sqlite3


@dataclass
class Company:
    """Company identifier"""
    cik: str # primary key
    name: str
    ticker: str
    sic: int
    industry: str


    def save(self, connection, *, commit: bool = True):
        connection.execute(
            """
            INSERT INTO company (cik, name, ticker, sic, industry)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cik) DO UPDATE SET
                name = excluded.name,
                ticker = excluded.ticker,
                sic = excluded.sic,
                industry = excluded.industry
            """,
            (self.cik, self.name, self.ticker, self.sic, self.industry)
        )
        if commit:
            connection.commit()
        logger.info(f"Saved Company cik={self.cik}, ticker={self.ticker}, name={self.name}")
        return self.cik


create_table_company = """
CREATE TABLE IF NOT EXISTS company (
    cik TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    ticker TEXT NOT NULL UNIQUE,
    sic INTEGER,
    industry TEXT
) STRICT;
"""
@dataclass
class CompanyFiling:
    """A single SEC 10-K or 10-Q report"""
    cik: str # foreign key
    accession_number: str # primary key
    form: str
    filing_date: str
    filing_year: int
    source_url: str
    fiscal_year: str # "FY2024"
    fiscal_period: str
    fiscal_period_start: str
    fiscal_period_end: str
    parser_version: str = ""
    parse_complete: bool = False
    
    def save(self, connection, *, commit: bool = True):
        connection.execute(
            """
            INSERT INTO filing (cik, accession_number, form, filing_date,
            filing_year, fiscal_year, fiscal_period, fiscal_period_start,
            fiscal_period_end, source_url, parser_version, parse_complete)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (self.cik, self.accession_number, self.form, self.filing_date,
            self.filing_year, self.fiscal_year, self.fiscal_period,
            self.fiscal_period_start, self.fiscal_period_end, self.source_url,
            self.parser_version, self.parse_complete)
        )
        if commit:
            connection.commit()
        logger.info(f"Saved CompanyFiling accession_number={self.accession_number}, cik={self.cik}")
        return self.accession_number

create_table_filing = """
CREATE TABLE IF NOT EXISTS filing (
    accession_number TEXT PRIMARY KEY,
    cik TEXT NOT NULL REFERENCES company(cik),
    form TEXT NOT NULL CHECK (form IN ('10-K', '10-Q')),
    filing_date TEXT NOT NULL,
    filing_year INTEGER NOT NULL,
    fiscal_year TEXT NOT NULL,
    fiscal_period TEXT NOT NULL,
    fiscal_period_start TEXT NOT NULL,
    fiscal_period_end TEXT NOT NULL,
    source_url TEXT NOT NULL,
    parser_version TEXT NOT NULL DEFAULT '',
    parse_complete INTEGER NOT NULL DEFAULT 0 CHECK (parse_complete IN (0, 1))
) STRICT;
"""

@dataclass
class FilingChunk:
    """A chunk of text from a company filing"""
    chunk_id: str
    accession_number: str
    part: str              # "PART I" or "PART II"
    item: str              # "ITEM 7"
    item_title: str        # "MANAGEMENT'S DISCUSSION..."
    chunk_index: int
    text: str
    parser_version: str
    def save(self, connection, *, commit: bool = True):
        connection.execute(
            """
            INSERT INTO filing_chunk (accession_number, chunk_id, part, item,
            item_title, chunk_index, text, parser_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (self.accession_number, self.chunk_id, self.part, self.item,
             self.item_title, self.chunk_index, self.text, self.parser_version)
        )
        if commit:
            connection.commit()
        logger.debug(f"Saved FilingChunk chunk_id={self.chunk_id}")
        return self.chunk_id

create_table_filing_chunk = """
CREATE TABLE IF NOT EXISTS filing_chunk (
    chunk_id TEXT PRIMARY KEY,
    accession_number TEXT NOT NULL REFERENCES filing(accession_number),
    part TEXT NOT NULL,
    item TEXT NOT NULL,
    item_title TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    parser_version TEXT NOT NULL
) STRICT;
"""

def sqlite_connection():
    connection = sqlite3.connect("data/corpus.sqlite")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def migrate_schema(connection) -> None:
    """Add columns introduced after an existing corpus database was created."""
    filing_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(filing)")
    }

    if "parser_version" not in filing_columns:
        connection.execute(
            "ALTER TABLE filing ADD COLUMN parser_version TEXT NOT NULL DEFAULT ''"
        )
    if "parse_complete" not in filing_columns:
        connection.execute(
            "ALTER TABLE filing ADD COLUMN parse_complete INTEGER NOT NULL DEFAULT 0"
        )

def setup_db() -> None:
    conn = sqlite_connection()
    conn.execute(create_table_company)
    conn.execute(create_table_filing)
    conn.execute(create_table_filing_chunk)
    migrate_schema(conn)
    conn.commit()

if __name__ == "__main__":
    setup_db()
