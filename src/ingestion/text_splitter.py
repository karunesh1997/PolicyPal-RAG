from typing import List, Dict


def split_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Split text into overlapping character chunks.
    """

    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def create_chunks(
    documents: List[Dict],
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[Dict]:
    """
    Split page-level documents into chunks while
    retaining source metadata.
    """

    chunks = []

    for document in documents:

        text_chunks = split_text(
            document["text"],
            chunk_size,
            chunk_overlap
        )

        for chunk_number, chunk in enumerate(
            text_chunks,
            start=1
        ):

            chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "page": document["page"],
                    "chunk": chunk_number,
                }
            )

    return chunks