import torch
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from .vector_store import QueryFilters
from .constants import CONTEXT_DIR
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
@dataclass
class HySTResult:
    query: str
    filters: QueryFilters

class HySTQuery:
    """
    HyST: LLM-Powered Hybrid Retrieval over Semi-Structured Tabular Data
    https://arxiv.org/abs/2508.18048

    HyST decomposes a user query into (1) structured filtering conditions and (2) unstructured semantic intent.
    """
    def __init__(self, model_name):
        self.prompt = (CONTEXT_DIR/"hyst_query.md").read_text()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Loading model {model_name}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,dtype=torch.float16
        ).to(device)
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = outlines.from_transformers(hf_model, hf_tokenizer)
        self.generator = outlines.Generator(model, QueryFilters)
        logger.info(f"HySTQuery initialized with {model_name}")

    def query_filters(self, question: str) -> HySTResult:
        prompt = f"{self.prompt}\n\nQuestion:\n{question}"
        try:
            raw = self.generator(prompt, max_new_tokens=150)
            logger.info(f"HyST Generator output: {raw}")
            query_filters = QueryFilters.model_validate_json(raw)
        except ValueError as e:
            logger.warning(f"HyST extraction failed for {question}: {e}")
            query_filters = QueryFilters()
        return HySTResult(query=question, filters=query_filters)
