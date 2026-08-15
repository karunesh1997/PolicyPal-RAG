from pathlib import Path
from typing import List, Dict

from config.config import (
    DOCUMENTS_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from src.ingestion.pdf_loader import (
    load_all_pdfs
)

from src.ingestion.text_splitter import (
    create_chunks
)

from src.embeddings.embedder import (
    Embedder
)

from src.vectorstore.faiss_store import (
    FAISSStore
)

from src.rag.retriever import (
    Retriever
)

from src.rag.generator import (
    Generator
)


class PolicyPipeline:

    def __init__(self):

        self.embedder = Embedder()

        self.vectorstore = FAISSStore(
            FAISS_INDEX_PATH,
            METADATA_PATH
        )

        self.retriever = Retriever(
            self.embedder,
            self.vectorstore
        )

        self.generator = Generator()

    def build_index(self) -> int:
        """
        Load PDFs, split them, embed them,
        and build FAISS index.

        Returns:
            Number of chunks indexed.
        """

        documents = load_all_pdfs(
            DOCUMENTS_DIR
        )

        if not documents:
            raise ValueError(
                "No PDF documents found."
            )

        chunks = create_chunks(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedder.encode(
            texts
        )

        self.vectorstore.build(
            embeddings,
            chunks
        )

        self.vectorstore.save()

        return len(chunks)

    def load_index(self):

        self.vectorstore.load()

    def ask(
        self,
        question: str,
        top_k: int = 5
    ):

        results = self.retriever.retrieve(
            question,
            top_k
        )

        answer = self.generator.answer(
            question,
            results
        )

        confidence = (
            self.retriever.average_score(
                results
            )
        )

        return {
            "answer": answer,
            "sources": results,
            "confidence": confidence
        }

    def is_ready(self) -> bool:

        return self.vectorstore.exists()