"""
This file contains the with_latex() and without_latex() functions, which return a dictionary of {id: text} for every row in a cleaned .jsonl file.
The with_latex() function includes the original formulas in the text, while the without_latex() function excludes the original formulas. 

Example usage:
    from src.preprocessing.latex2sem import with_latex
    answers_with_latex = with_latex("data/clean/answers.jsonl")
"""

import json
from tqdm import tqdm
from pathlib import Path

from src.utils import splice, get_text


def with_latex(source: Path) -> dict:
    """
    Given a cleaned ``.jsonl`` file, returns ``{id: text}`` for every row in it, including original formulas.
    """
    results = {}
    with open(source, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            row = json.loads(line)
            row_id = row.pop("id")
            results[row_id] = splice(row.pop("body"))
    return results


def without_latex(source: Path) -> dict:
    """
    Given a cleaned ``.jsonl`` file, returns ``{id: text}`` for every row in it, excluding original formulas.
    """
    results = {}
    with open(source, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            row = json.loads(line)
            row_id = row.pop("id")
            text = get_text(row.pop("body"))
            results[row_id] = " ".join(text).strip()
    return results
