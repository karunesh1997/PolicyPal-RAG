from pathlib import Path
from typing import List


def get_pdf_files(
    directory: str | Path
) -> List[Path]:

    directory = Path(directory)

    return sorted(
        directory.glob("*.pdf")
    )


def format_source(
    source: str,
    page: int
) -> str:

    return f"{source} — Page {page}"


def confidence_label(
    score: float
) -> str:

    if score >= 0.75:
        return "High"

    if score >= 0.50:
        return "Medium"

    return "Low"