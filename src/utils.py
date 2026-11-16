"""Read the cleaned data written by ``src.preprocessing.clean``.

A field is a list of pieces. Text is a ``str`` and a formula is a dict:

    ["Let ", {"latex": "x^2"}, " be even."]

``splice`` joins pieces back into one string. ``load`` reads a cleaned
``.jsonl`` file, keeps formulas when asked, and appends a translation file
when one is given.
"""

import json
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

Piece = str | dict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"

CLEANED_DIR = DATA_DIR / "clean"
CLEANED_ANSWERS = CLEANED_DIR / "answers.jsonl"

TOPIC_YEARS = (2020, 2021, 2022)
CLEANED_TOPICS = [CLEANED_DIR / f"topics_{year}.jsonl" for year in TOPIC_YEARS]
QRELS = {year: DATA_DIR / "qrels" / f"task1_{year}.tsv" for year in TOPIC_YEARS}

TRANSLATION_DIR = DATA_DIR / "translations"

INDEX_DIR = DATA_DIR / "indexes"
INDEX_MAP = {
    "by_detex": INDEX_DIR / "by_detex_index.json",
    "by_llm": INDEX_DIR / "by_llm_index.json",
    "without_context": INDEX_DIR / "without_context_index.json",
    "with_context": INDEX_DIR / "with_context_index.json",
    "with_latex": INDEX_DIR / "with_latex_index.json",
    "without_latex": INDEX_DIR / "without_latex_index.json",
    "neural_pt": INDEX_DIR / "neural_pt_index.npz",
    "neural_ft": INDEX_DIR / "neural_ft_index.npz"
}

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


def load(source: str | Path = CLEANED_ANSWERS, translations: str | Path | None = None, with_latex: bool = False) -> dict[str, str]:
    """
    Read a cleaned file into ``{id: text}``.

    ``source`` defaults to ``data/clean/answers.jsonl``. List fields are read
    in order, so a topic contributes its title and then its question.
    Formulas stay as ``$latex$`` when ``with_latex`` is set, and are dropped
    otherwise. ``translations`` maps a document id to text that is appended.
    With no translation file, each document keeps its own text.
    """
    saved = {}
    if translations is not None:
        if with_latex:
            print("WARNING: with_latex argument ignored.\nOnly non-translated data can be loaded with original LaTeX.")
            with_latex = False
        path = Path(translations)
        if not path.exists():
            raise FileNotFoundError(f"Translation file {path} not found.")
        with open(path, encoding="utf-8") as file:
            saved = json.load(file)

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Cleaned file {source} not found.")
    texts = {}
    with open(source, encoding="utf-8") as file:
        for line in tqdm(file, desc=source.name):
            row = json.loads(line)
            doc_id = str(row.pop("id"))
            pieces = [piece for field in row.values() if isinstance(field, list) for piece in field]
            text = splice(pieces) if with_latex else " ".join(get_text(pieces)).strip()
            extra = saved.get(doc_id)
            if extra and str(extra).strip():
                text = f"{text} {str(extra).strip()}".strip()
            texts[doc_id] = text
    return texts
