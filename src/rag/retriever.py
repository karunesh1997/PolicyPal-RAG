from typing import List, Dict

from config.config import (
    TOP_K,
    MIN_SIMILARITY
)


class Retriever:

    def __init__(
        self,
        embedder,
        vectorstore
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a question.
        """

        query_embedding = (
            self.embedder.encode_query(query)
        )

        results = self.vectorstore.search(
            query_embedding,
            top_k=top_k
        )

        filtered_results = []

        for metadata, score in results:

            if score >= MIN_SIMILARITY:

                result = metadata.copy()

                result["score"] = round(
                    score,
                    4
                )

                filtered_results.append(
                    result
                )

        return filtered_results

    @staticmethod
    def average_score(
        results: List[Dict]
    ) -> float:

        if not results:
            return 0.0

        return sum(
            result["score"]
            for result in results
        ) / len(results)