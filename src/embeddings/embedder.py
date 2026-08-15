from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import EMBEDDING_MODEL


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def encode(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Convert text into normalized embeddings.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings.astype("float32")

    def encode_query(
        self,
        query: str
    ) -> np.ndarray:
        """
        Encode a single query.
        """

        embedding = self.encode([query])

        return embedding[0]