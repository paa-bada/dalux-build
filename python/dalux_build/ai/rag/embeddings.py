"""LangChain embedding interface backed by the Mistral SDK."""

import math
import os

from langchain_core.embeddings import Embeddings
from mistralai.client import Mistral

from .vectorstore import DEFAULT_EMBEDDING_MODEL


class MistralEmbeddings(Embeddings):
    """Batch document chunks and queries through mistral-embed."""

    def __init__(self) -> None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY must be set for embeddings.")
        self.client = Mistral(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 96):
            batch = [text[:8000] for text in texts[start : start + 96]]
            response = self.client.embeddings.create(model=DEFAULT_EMBEDDING_MODEL, inputs=batch)
            items = sorted(response.data, key=lambda item: item.index if item.index is not None else -1)
            if len(items) != len(batch):
                raise ValueError("Mistral returned incomplete embeddings.")
            for index, item in enumerate(items):
                if (
                    item.index != index
                    or not item.embedding
                    or not all(math.isfinite(value) for value in item.embedding)
                ):
                    raise ValueError("Mistral returned invalid embeddings.")
                vectors.append(item.embedding)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
