"""Split post and topic HTML into text and formulas, and splice text back.

Every approach runs:

    pieces = split_formulas(html)
    text = splice(pieces, translate)

``translate`` takes one ``Formula`` and returns its replacement, or None to
put the original formula back.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from lxml import etree
from lxml import html as lxml_html

_WHITESPACE = re.compile(r"\s+")
# Control characters that lxml rejects in text, written directly or as ``&#1;``.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NUMERIC_ENTITY = re.compile(r"&#(?:[xX]([0-9a-fA-F]+)|([0-9]+));")
_BLOCK_TAGS = {"p", "div", "li", "blockquote", "h1", "h2", "h3", "h4", "pre", "tr"}
_IS_MATH = "contains(concat(' ', normalize-space(@class), ' '), ' math-container ')"
_OUTER_SPANS = etree.XPath(f"//span[{_IS_MATH}][not(ancestor::span[{_IS_MATH}])]")

# Private-use characters mark where each math-container sat in the text.
_SPAN_MARK = re.compile("\ue000(\\d+)\ue001")
_LATEX_CHARS = re.compile(r"[\\^_{}]")
_MATH_NAME = re.compile(r"[A-Za-z][0-9]*'*")
# Three or more plain words in a row is prose, as in "$5 and $10 for a ticket$".
_PROSE_RUN = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){2,}")


@dataclass(frozen=True)
class Formula:
    """One formula in a field. ``position`` counts formulas from 0 in that field."""

    latex: str
    display: bool
    position: int

    def source(self) -> str:
        delim = "$$" if self.display else "$"
        return f"{delim}{self.latex}{delim}"


Piece = str | Formula


def split_formulas(html: str | None) -> list[Piece]:
    """Split one field of post or topic HTML into text and ``Formula`` pieces.

    Each ``math-container`` is one formula, with one outer ``$`` or ``$$``
    layer removed. Text between containers is then scanned for ``$...$`` and
    ``$$...$$`` pairs that look like math. Any other ``$`` stays as text.
    """
    if not html or not html.strip():
        return []

    html = _NUMERIC_ENTITY.sub(_blank_control_entity, _CONTROL.sub(" ", html))
    spans = []
    try:
        root = lxml_html.fragment_fromstring(html, create_parent="div")
    except (etree.ParserError, ValueError):
        text = _WHITESPACE.sub(" ", html).strip()
    else:
        for span in _OUTER_SPANS(root):
            spans.append(_strip_delims(span.text_content()))
            _replace_with_text(span, f"\ue000{len(spans) - 1}\ue001")
        for el in list(root.iter()):
            if not isinstance(el.tag, str):
                continue
            if el.tag == "br":
                _replace_with_text(el, " ")
            elif el.tag in _BLOCK_TAGS:
                _append_text(el, " ")
        text = _WHITESPACE.sub(" ", root.text_content()).strip()

    raw = []
    start = 0
    for mark in _SPAN_MARK.finditer(text):
        raw.extend(_split_dollars(text[start:mark.start()]))
        latex, display = spans[int(mark.group(1))]
        if latex:
            raw.append((latex, display))
        start = mark.end()
    raw.extend(_split_dollars(text[start:]))

    pieces: list[Piece] = []
    for item in raw:
        if isinstance(item, str):
            if pieces and isinstance(pieces[-1], str):
                pieces[-1] += item
            elif item:
                pieces.append(item)
        else:
            count = sum(isinstance(p, Formula) for p in pieces)
            pieces.append(Formula(item[0], item[1], count))
    return pieces


def formulas_in(pieces: list[Piece]) -> list[Formula]:
    return [p for p in pieces if isinstance(p, Formula)]


def splice(pieces: list[Piece], translate: Callable[[Formula], str | None] | None = None) -> str:
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
            parts.append(piece.source())
    return _WHITESPACE.sub(" ", "".join(parts)).strip()


def encode_pieces(pieces: list[Piece]) -> list[str | dict]:
    """Pieces as JSON values: text stays a string, a formula becomes a dict."""
    return [p if isinstance(p, str) else {"latex": p.latex, "display": p.display} for p in pieces]


def decode_pieces(items: list[str | dict]) -> list[Piece]:
    pieces: list[Piece] = []
    position = 0
    for item in items:
        if isinstance(item, str):
            pieces.append(item)
        else:
            pieces.append(Formula(item["latex"], item["display"], position))
            position += 1
    return pieces


def _blank_control_entity(match: re.Match) -> str:
    code = int(match.group(1), 16) if match.group(1) else int(match.group(2))
    if code > 0x10FFFF or _CONTROL.match(chr(code)):
        return " "
    return match.group(0)


def _replace_with_text(el, text: str) -> None:
    """Remove ``el`` and its children, leaving ``text`` and ``el``'s tail in its place."""
    text += el.tail or ""
    prev = el.getprevious()
    parent = el.getparent()
    if prev is not None:
        prev.tail = (prev.tail or "") + text
    else:
        parent.text = (parent.text or "") + text
    parent.remove(el)


def _append_text(el, text: str) -> None:
    children = list(el)
    if children:
        children[-1].tail = (children[-1].tail or "") + text
    else:
        el.text = (el.text or "") + text


def _strip_delims(text: str) -> tuple[str, bool]:
    s = text.strip()
    if not s.strip("$"):
        return "", False
    if len(s) >= 4 and s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip(), True
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip(), False
    return s, False


def _is_open(text: str, i: int) -> bool:
    return text[i] == "$" and (i == 0 or text[i - 1] != "\\")


def _next_dollar(text: str, start: int) -> int:
    i = text.find("$", start)
    while i != -1 and not _is_open(text, i):
        i = text.find("$", i + 1)
    return i


def _looks_like_math(inner: str) -> bool:
    s = inner.strip()
    if not s or _PROSE_RUN.search(s):
        return False
    return bool(_LATEX_CHARS.search(s) or _MATH_NAME.fullmatch(s))


def _split_dollars(text: str) -> list[str | tuple[str, bool]]:
    out: list[str | tuple[str, bool]] = []
    start = 0
    i = 0
    while True:
        j = _next_dollar(text, i)
        if j == -1:
            break
        display = text.startswith("$$", j)
        delim = "$$" if display else "$"
        open_end = j + len(delim)
        k = text.find(delim, open_end)
        while k != -1 and not _is_open(text, k):
            k = text.find(delim, k + 1)
        if k != -1 and _looks_like_math(text[open_end:k]):
            out.append(text[start:j])
            out.append((text[open_end:k].strip(), display))
            start = i = k + len(delim)
        else:
            i = open_end
    out.append(text[start:])
    return out
