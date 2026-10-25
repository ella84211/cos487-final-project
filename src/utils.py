"""Get formulas from the cleaned data written by ``src.preprocessing.clean``, and join them back with text.

A field is a list of pieces. Text is a ``str`` and a formula is a dict:

    ["Let ", {"latex": "x^2"}, " be even."]
"""

from collections.abc import Callable

Piece = str | dict


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

    For use with the cleaned data written by ``src/preprocessing/clean.py``.
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
