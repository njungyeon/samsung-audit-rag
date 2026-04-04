from __future__ import annotations

import sys

from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1 and sys.stdout.isatty(),
        )
        return vectors.tolist()
