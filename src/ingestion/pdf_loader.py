from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> List[Dict]:
    """
    Load a PDF and return page-level text.

    Returns:
        [
            {
                "text": "...",
                "source": "Leave_Policy.pdf",
                "page": 1
            }
        ]
    """

    file_path = Path(file_path)

    reader = PdfReader(str(file_path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "source": file_path.name,
                "page": page_number,
            }
        )

    return documents


def load_all_pdfs(directory: str | Path) -> List[Dict]:
    """
    Load all PDF files from a directory.
    """

    directory = Path(directory)

    all_documents = []

    pdf_files = sorted(directory.glob("*.pdf"))

    for pdf_file in pdf_files:
        try:
            pages = load_pdf(pdf_file)
            all_documents.extend(pages)

        except Exception as exc:
            print(
                f"Failed to process {pdf_file.name}: {exc}"
            )

    return all_documents