import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = BASE_DIR / "data"

DOCUMENTS_DIR = DATA_DIR / "documents"

VECTORSTORE_DIR = DATA_DIR / "vectorstore"


DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Gemini
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Chunking
CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "700"
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "100"
    )
)


# Retrieval
TOP_K = int(
    os.getenv(
        "TOP_K",
        "5"
    )
)

MIN_SIMILARITY = float(
    os.getenv(
        "MIN_SIMILARITY",
        "0.30"
    )
)


FAISS_INDEX_PATH = (
    VECTORSTORE_DIR / "index.faiss"
)

METADATA_PATH = (
    VECTORSTORE_DIR / "metadata.json"
)