"""Read ARQMath answers and Task 1 topics, raw or cleaned.

``src/preprocessing/clean.py`` reads the raw files once and writes each
answer and topic as text and formula pieces to ``data/clean/``:

    for answer_id, pieces in iter_clean_answers():
        ...
"""

import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from src.utils.formulas import Piece, decode_pieces

_ARQ_ROOT = Path(__file__).resolve().parents[2] / "ARQMathCode"
if str(_ARQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_ARQ_ROOT))

from topic_file_reader import TopicReader

POSTS = Path("data/Posts.V1.3.xml")
TOPIC_YEARS = (2020, 2021, 2022)
CLEAN_DIR = Path("data/clean")
CLEAN_ANSWERS = CLEAN_DIR / "answers.jsonl"


def topics_path(year: int) -> Path:
    return Path(f"data/topics/Topics_Task1_{year}.xml")


def clean_topics_path(year: int) -> Path:
    return CLEAN_DIR / f"topics_{year}.jsonl"


def iter_answers(posts_path: str | Path = POSTS) -> Iterator[tuple[int, str]]:
    """Yield ``(answer id, body HTML)`` for every answer, one row at a time."""
    context = ET.iterparse(posts_path, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event != "end" or elem.tag != "row":
            continue
        if elem.get("PostTypeId") == "2":
            yield int(elem.get("Id")), elem.get("Body") or ""
        root.clear()


def iter_topics(topic_path: str | Path) -> Iterator[tuple[str, str, str]]:
    """Yield ``(topic id, title HTML, question HTML)`` for a Task 1 topic file."""
    for topic_id, topic in TopicReader(str(topic_path)).map_topics.items():
        yield topic_id, topic.title or "", topic.question or ""


def iter_clean_answers(path: str | Path = CLEAN_ANSWERS) -> Iterator[tuple[int, list[Piece]]]:
    """Yield ``(answer id, pieces)`` from the file ``clean.py`` wrote."""
    with open(_require(path), encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            yield row["id"], decode_pieces(row["body"])


def iter_clean_topics(year: int) -> Iterator[tuple[str, list[Piece], list[Piece]]]:
    """Yield ``(topic id, title pieces, question pieces)`` for one Task 1 year."""
    with open(_require(clean_topics_path(year)), encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            yield row["id"], decode_pieces(row["title"]), decode_pieces(row["question"])


def _require(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.preprocessing.clean")
    return path
