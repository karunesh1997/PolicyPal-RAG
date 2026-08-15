from src.ingestion.text_splitter import (
    split_text,
    create_chunks
)


def test_split_text():

    text = "A" * 1500

    chunks = split_text(
        text,
        chunk_size=500,
        chunk_overlap=50
    )

    assert len(chunks) > 1


def test_create_chunks():

    documents = [
        {
            "text": "Employee leave policy information.",
            "source": "leave.pdf",
            "page": 2
        }
    ]

    chunks = create_chunks(
        documents,
        chunk_size=100,
        chunk_overlap=20
    )

    assert len(chunks) >= 1

    assert chunks[0]["source"] == "leave.pdf"

    assert chunks[0]["page"] == 2