from pydantic import BaseModel, field_validator, ConfigDict
from typing import ClassVar

class QueryFilters(BaseModel):
    ticker: dict[str, str] = {}
    doc_type: dict[str, str] = {}
    year: dict[str, int | list[int]] = {}

    model_config = ConfigDict(extra='forbid')
    
    OPERATORS: ClassVar[set[str]] = {"$eq", "$in", "$lt", "$gt", "$lte", "$gte"}
    TICKERS: ClassVar[set[str]] = {'XOM', 'MRK', 'GM', 'TSLA', 'LLY', 'AMGN', 'MRNA', 'JNJ', 'EPD', 'ET', 'WMB', 'F', 'LNG', 'KMI', 'ABBV', 'EOG', 'COP', 'CTRA', 'CVX', 'MPC', 'DVN', 'PFE'}
    DOC_TYPES: ClassVar[set[str]] = {"10-Q", "10-K"}
    YEARS: ClassVar[set[int]] = {2024, 2025}

    @field_validator("ticker", "doc_type", "year")
    @classmethod
    def validate_operator(cls, value: dict) -> dict:
        for k in list(value.keys()):
            if k not in cls.OPERATORS:
                value.pop(k)
        return value

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: dict) -> dict:  
        return cls._validate_in(value, cls.TICKERS)

    @field_validator("doc_type")
    @classmethod
    def validate_doc_type(cls, value: dict) -> dict:  
        return cls._validate_in(value, cls.DOC_TYPES)

    @field_validator("year")
    @classmethod
    def validate_year(cls, value: dict) -> dict:  
        return cls._validate_in(value, cls.YEARS)

    @classmethod
    def _validate_in(cls, value: dict, class_set: set) -> dict:
        for k in list(value.keys()):
            v = value[k]
            if isinstance(v, list):
                vals = [val for val in v if val in class_set]
                if vals:
                    value[k] = vals
                else:
                    value.pop(k)
            elif v not in class_set:
                value.pop(k)
        return value
