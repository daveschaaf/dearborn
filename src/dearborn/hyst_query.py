import torch
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from .vector_store import QueryFilters
from .constants import CONTEXT_DIR
from dataclasses import dataclass

@dataclass
class HySTResult:
    question: str
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
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,dtype=torch.float16
        ).to(device)
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = outlines.from_transformers(hf_model, hf_tokenizer)
        self.generator = outlines.Generator(model, QueryFilters)

    def query_filters(self, question: str) -> HySTResult:
        prompt = f"{self.prompt}\n\nQuestion:\n{question}"
        query_filters = self.generator(prompt, max_new_tokens=150)
        return HySTResult(question=question, filters=query_filters)

