"""
This script splits every answer and Task 1 topic into text and formula pieces, and writes
them to data/clean/. Every preprocessing approach and the indexer read these files, so they
all see the same formulas in the same order.

Each line of data/clean/answers.jsonl is one answer:
    {"id": 2190525, "body": ["Let ", {"latex": "x^2", "display": false}, " be even."]}
Each line of data/clean/topics_<year>.jsonl is one topic, with "id", "title", and "question".

It should be run as a module from the root of the repository. Assumes the /data/ directory
contains Posts.V1.3.xml and the Task 1 topic files.

Example usage:
    python -m src.preprocessing.clean
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from src.utils.arqmath import (
    CLEAN_ANSWERS,
    POSTS,
    TOPIC_YEARS,
    clean_topics_path,
    iter_answers,
    iter_topics,
    topics_path,
)
from src.utils.formulas import encode_pieces, split_formulas


def _write_lines(path: Path, rows) -> int:
    """Write rows as JSON lines through a temporary file, so a stopped run leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    try:
        with open(tmp, "w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return count


def clean_topics() -> None:
    for year in TOPIC_YEARS:
        rows = (
            {"id": topic_id, "title": encode_pieces(split_formulas(title)),
             "question": encode_pieces(split_formulas(question))}
            for topic_id, title, question in iter_topics(topics_path(year))
        )
        count = _write_lines(clean_topics_path(year), rows)
        print(f"{count} topics saved to {clean_topics_path(year)}.")


def clean_answers(posts_path: str) -> None:
    rows = (
        {"id": answer_id, "body": encode_pieces(split_formulas(body))}
        for answer_id, body in tqdm(iter_answers(posts_path), desc="answers")
    )
    count = _write_lines(CLEAN_ANSWERS, rows)
    print(f"{count} answers saved to {CLEAN_ANSWERS}.")


def main():
    parser = argparse.ArgumentParser(description="Split ARQMath answers and topics into text and formula pieces.")
    parser.add_argument("--posts_file", type=str, default=str(POSTS), help="Path to Posts.V1.3.xml.")
    parser.add_argument("--topics_only", action="store_true", help="Only rewrite the topic files.")
    args = parser.parse_args()

    clean_topics()
    if not args.topics_only:
        clean_answers(args.posts_file)


if __name__ == "__main__":
    main()
