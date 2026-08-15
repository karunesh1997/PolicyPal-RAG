import json
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np


class FAISSStore:

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index = None
        self.metadata = []

    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Build a FAISS cosine-similarity index.
        """

        if len(embeddings) == 0:
            raise ValueError(
                "No embeddings supplied."
            )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings)

        self.index = index
        self.metadata = metadata

    def save(self):
        """
        Save FAISS index and metadata.
        """

        if self.index is None:
            raise ValueError(
                "Cannot save an empty index."
            )

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self):
        """
        Load existing FAISS index and metadata.
        """

        if not self.index_path.exists():
            raise FileNotFoundError(
                "FAISS index does not exist."
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                "Metadata file does not exist."
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """
        Search the vector store.

        Returns:
            [(metadata, similarity_score), ...]
        """

        if self.index is None:
            raise ValueError(
                "Vector store has not been loaded."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            if index >= len(self.metadata):
                continue

            results.append(
                (
                    self.metadata[index],
                    float(score)
                )
            )

        return results

    def exists(self) -> bool:
        return (
            self.index_path.exists()
            and self.metadata_path.exists()
        )