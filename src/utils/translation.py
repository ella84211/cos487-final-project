"""Translate formulas for one approach and produce the text the indexer reads.

A ``Translation`` wraps one function that translates a single formula. Every
result is appended to a cache file as soon as it is made, so a stopped run
picks up where it left off and a finished run never translates a formula twice.

    translation = Translation(translate_one, "data/latex2eng/llm.jsonl")
    for answer_id, text in translation.answers():
        ...
    for topic_id, title, question in translation.topics(2020):
        ...
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

from src.utils.arqmath import CLEAN_ANSWERS, iter_clean_answers, iter_clean_topics
from src.utils.formulas import Formula, splice


class Translation:
    def __init__(self, translate_one: Callable[[str], str | None], cache_path: str | Path):
        """``translate_one`` takes a formula's LaTeX and returns its translation, or None."""
        self._translate_one = translate_one
        self.cache_path = Path(cache_path)
        self._cache: dict[str, str | None] = {}
        self._file = None
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as file:
                for line in file:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._cache[row["latex"]] = row["text"]

    def __call__(self, formula: Formula) -> str | None:
        """Return the translation of ``formula``, or None to put the formula back."""
        latex = formula.latex
        if latex in self._cache:
            return self._cache[latex]
        text = self._translate_one(latex) or None
        self._cache[latex] = text
        self._append(latex, text)
        return text

    def answers(self, path: str | Path = CLEAN_ANSWERS) -> Iterator[tuple[int, str]]:
        """Yield ``(answer id, translated text)`` for every cleaned answer."""
        try:
            for answer_id, pieces in iter_clean_answers(path):
                yield answer_id, splice(pieces, self)
        finally:
            self.close()

    def topics(self, year: int) -> Iterator[tuple[str, str, str]]:
        """Yield ``(topic id, translated title, translated question)`` for one Task 1 year."""
        try:
            for topic_id, title, question in iter_clean_topics(year):
                yield topic_id, splice(title, self), splice(question, self)
        finally:
            self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _append(self, latex: str, text: str | None) -> None:
        if self._file is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # A run killed mid-write leaves a partial last line; start on a fresh one.
            partial = False
            if self.cache_path.exists() and self.cache_path.stat().st_size > 0:
                with open(self.cache_path, "rb") as file:
                    file.seek(-1, 2)
                    partial = file.read(1) != b"\n"
            self._file = open(self.cache_path, "a", encoding="utf-8")
            if partial:
                self._file.write("\n")
        self._file.write(json.dumps({"latex": latex, "text": text}, ensure_ascii=False) + "\n")
        self._file.flush()
