"""Read the cleaned data written by ``src.preprocessing.clean``.

A field is a list of pieces. Text is a ``str`` and a formula is a dict:

    ["Let ", {"latex": "x^2"}, " be even."]

``splice`` joins pieces back into one string. ``load`` drops the formulas
from ``data/clean/answers.jsonl`` and appends a saved translation file.
"""

import json
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

Piece = str | dict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLEANED_DIR = DATA_DIR / "clean"
CLEANED_ANSWERS = CLEANED_DIR / "answers.jsonl"

TRANSLATION_DIR = DATA_DIR / "translations"

CLEANED_ANSWERS = CLEANED_DIR / "answers.jsonl"

TOPIC_YEARS = (2020, 2021, 2022)
CLEANED_TOPICS = [CLEANED_DIR / f"topics_{year}.jsonl" for year in TOPIC_YEARS]


def get_formulas(pieces: list[Piece]) -> list[dict]:
    """Return the formula pieces of one field, in order.

    For use with the cleaned data written by ``src/preprocessing/clean.py``.
    """
    return [p for p in pieces if isinstance(p, dict)]


def get_text(pieces: list[Piece]) -> list[str]:
    """Return the text pieces of one field, in order, with no formulas.

    For use with the cleaned data written by ``src/preprocessing/clean.py``.
    """
    return [p for p in pieces if isinstance(p, str)]


def splice(pieces: list[Piece], translate: Callable[[dict], str | None] | None = None) -> str:
    """Join pieces back into text, replacing each formula with ``translate``.

    A formula whose translation is None or blank is put back as ``$latex$``.
    With no ``translate``, every formula is put back.
    """
    parts = []
    for piece in pieces:
        if isinstance(piece, str):
            parts.append(piece)
            continue
        replacement = translate(piece) if translate else None
        if replacement and replacement.strip():
            parts.append(replacement.strip())
        else:
            parts.append(f"${piece['latex']}$")
    return " ".join("".join(parts).split())


def load(translations: str | Path) -> dict[str, str]:
    """
    Drop the formulas from the cleaned answers and append the saved translations.

    Reads cleaned answers from ``data/clean/answers.jsonl``. 
    ``translations`` maps a document id to the text to append. 
    A document with no saved translation keeps its text alone.
    """
    path = Path(translations)
    if not path.exists():
        raise FileNotFoundError(f"Translation file {path} not found.")
    if not CLEANED_ANSWERS.exists():
        raise FileNotFoundError(f"Cleaned answers file {CLEANED_ANSWERS} not found.")
    with open(path, encoding="utf-8") as file:
        saved = json.load(file)

    texts = {}
    with open(CLEANED_ANSWERS, encoding="utf-8") as file:
        for line in tqdm(file, desc=CLEANED_ANSWERS.name):
            row = json.loads(line)
            doc_id = str(row.pop("id"))
            pieces = [piece for field in row.values() if isinstance(field, list) for piece in field]
            text = " ".join(get_text(pieces)).strip()
            extra = saved.get(doc_id)
            if extra and str(extra).strip():
                text = f"{text} {str(extra).strip()}".strip()
            texts[doc_id] = text
    return texts
