from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

VERSION = "e5-mistral-section-paragraph-v1-target-384-overlap-1p"
class MistralRetriever:
    MODEL_NAME = "intfload/e5-mistral-7b-instruct"
    def __init__(self):
        self.version = VERSION
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.max_tokens = 4096
        self.chunk_size = 300


