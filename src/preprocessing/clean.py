"""
This script splits every answer and Task 1 topic into text and formula pieces, and writes
them to data/clean/. Every preprocessing approach reads these files, so they all see the
same formulas in the same order.

Each line of data/clean/answers.jsonl is one answer:
    {"id": 2190525, "body": ["Let ", {"latex": "x^2"}, " be even."]}
Each line of data/clean/topics_<year>.jsonl is one topic, with "id", "title", and "question".

It should be run as a module from the root of the repository. Assumes the /data/ directory
contains Posts.V1.3.xml and the Task 1 topic files.

Example usage:
    python -m src.preprocessing.clean            # answers and topics
    python -m src.preprocessing.clean topics     # only topics
    python -m src.preprocessing.clean answers    # only answers
"""

import argparse
import json
import re
import sys
from pathlib import Path

from lxml import etree
from lxml import html as lxml_html
from tqdm import tqdm

from src.utils import Piece

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ARQMathCode"))
from Entity_Parser_Record.post_parser_record import PostParserRecord
from topic_file_reader import TopicReader


POSTS = Path("data/Posts.V1.3.xml")
TOPIC_YEARS = (2020, 2021, 2022)
CLEAN_DIR = Path("data/clean")

# Control characters that lxml rejects in text, written directly or as ``&#1;``.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NUMERIC_ENTITY = re.compile(r"&#(?:[xX]([0-9a-fA-F]+)|([0-9]+));")
_BLOCK_TAGS = {"br", "p", "div", "li", "blockquote", "h1", "h2", "h3", "h4", "pre", "tr"}
_OUTER_SPANS = etree.XPath("//span[@class='math-container'][not(ancestor::span[@class='math-container'])]")

# Private-use characters mark where each math-container sat in the text.
_SPAN_MARK = re.compile("\ue000(\\d+)\ue001")
_OPEN_DOLLAR = re.compile(r"(?<!\\)\$")
# A ``$...$`` or ``$$...$$`` pair starting at an opening ``$``. An unclosed ``$$`` is never retried as ``$``.
_DOLLAR_PAIR = re.compile(r"(\$\$?+)(.*?)(?<!\\)\1", re.DOTALL)
_LATEX_CHARS = re.compile(r"[\\^_{}]")
_MATH_NAME = re.compile(r"[A-Za-z][0-9]*'*")
# Three or more plain words in a row is prose, as in "$5 and $10 for a ticket$".
_PROSE_RUN = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){2,}")


def clean_answers() -> None:
    print(f"Reading {POSTS}...")
    answers = PostParserRecord(str(POSTS)).map_just_answers
    rows = ({"id": i, "body": split_formulas(a.body)} for i, a in tqdm(answers.items(), desc="answers"))
    _write_lines(CLEAN_DIR / "answers.jsonl", rows)


def clean_topics() -> None:
    for year in TOPIC_YEARS:
        topics = TopicReader(f"data/topics/Topics_Task1_{year}.xml").map_topics
        rows = (
            {"id": i, "title": split_formulas(t.title), "question": split_formulas(t.question)}
            for i, t in topics.items()
        )
        _write_lines(CLEAN_DIR / f"topics_{year}.jsonl", rows)


def split_formulas(html: str) -> list[Piece]:
    """Split one field of post or topic HTML into text and formula pieces.

    Each ``math-container`` is one formula, with one outer ``$`` or ``$$``
    layer removed. Text between containers is then scanned for ``$...$`` and
    ``$$...$$`` pairs that look like math. Any other ``$`` stays as text.
    """
    html = _NUMERIC_ENTITY.sub(_blank_control_entity, _CONTROL.sub(" ", html))
    root = lxml_html.fragment_fromstring(html, create_parent="div")
    spans = []
    for span in _OUTER_SPANS(root):
        spans.append(_strip_delims(span.text_content()))
        tail = span.tail
        span.clear()
        span.text, span.tail = f"\ue000{len(spans) - 1}\ue001", tail
    for el in root.iter(*_BLOCK_TAGS):
        el.tail = " " + (el.tail or "")
    text = " ".join(root.text_content().split())

    pieces = []
    start = 0
    for mark in _SPAN_MARK.finditer(text):
        _add_dollar_pairs(pieces, text[start:mark.start()])
        latex = spans[int(mark.group(1))]
        if latex:
            pieces.append({"latex": latex})
        start = mark.end()
    _add_dollar_pairs(pieces, text[start:])
    return pieces


def _add_dollar_pairs(pieces: list[Piece], text: str) -> None:
    """Append ``text`` to ``pieces``, with each math-looking ``$`` pair as a formula."""
    start = pos = 0
    while dollar := _OPEN_DOLLAR.search(text, pos):
        match = _DOLLAR_PAIR.match(text, dollar.start())
        if match and _is_formula(match.group(2)):
            _add_text(pieces, text[start:match.start()])
            pieces.append({"latex": match.group(2).strip()})
            start = pos = match.end()
        else:
            pos = dollar.start() + (2 if text.startswith("$$", dollar.start()) else 1)
    _add_text(pieces, text[start:])


def _add_text(pieces: list[Piece], text: str) -> None:
    if not text:
        return
    if pieces and isinstance(pieces[-1], str):
        pieces[-1] += text
    else:
        pieces.append(text)


def _is_formula(inner: str) -> bool:
    """Whether the text inside a ``$`` pair is a formula, as opposed to prose between two dollar amounts."""
    s = inner.strip()
    if not s or _PROSE_RUN.search(s):
        return False
    return bool(_LATEX_CHARS.search(s) or _MATH_NAME.fullmatch(s))


def _strip_delims(text: str) -> str:
    s = text.strip()
    if not s.strip("$"):
        return ""
    if len(s) >= 4 and s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip()
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip()
    return s


def _blank_control_entity(match: re.Match) -> str:
    code = int(match.group(1), 16) if match.group(1) else int(match.group(2))
    if code > 0x10FFFF or _CONTROL.match(chr(code)):
        return " "
    return match.group(0)


def _write_lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"{count} rows saved to {path}.")


def main():
    parser = argparse.ArgumentParser(description="Split ARQMath answers and topics into text and formula pieces.")
    parser.add_argument("target", nargs="?", choices=["answers", "topics"], help="Clean only this. Defaults to both.")
    args = parser.parse_args()

    try:
        if args.target in (None, "topics"):
            clean_topics()
        if args.target in (None, "answers"):
            clean_answers()
    except FileNotFoundError as e:
        sys.exit(f"{e.filename} not found. Check that the ARQMath data is installed correctly by running bin/install.sh.")


if __name__ == "__main__":
    main()
