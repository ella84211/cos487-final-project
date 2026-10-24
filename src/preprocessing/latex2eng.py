"""
This script uses Gemma to translate the LaTeX formulas in answers and topics into English.
Formulas are read from data/clean/, written by src.preprocessing.clean, so every approach
sees the same formulas.
The results are saved to a JSON file, specified by the --output_file argument, that maps
each formula's LaTeX to its English translation. A formula with no usable translation is
not saved, so splice puts the original formula back in its place.
Rerunning with the same output file skips formulas that are already translated.
It should be run as a module from the root of the repository, after src.preprocessing.clean.

Example usage:
    python -m src.preprocessing.latex2eng --output_file data/latex2eng.json
"""

import argparse
import json
import re
import sys
import types
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from src.utils.arqmath import TOPIC_YEARS, iter_clean_answers, iter_clean_topics
from src.utils.formulas import Formula, formulas_in


# Formulas longer than this many characters are not sent to the model.
TRUNCATE_LENGTH = 1000
_WHITESPACE = re.compile(r"\s+")

MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a skilled mathematician. Translate the LaTeX formula into natural language. "
            "For example, if the formula is x^2 + y^2 = z^2, the translation is 'x squared plus y squared equals z squared'. "
            "The result will be used in bag-of-words retrieval, so the translation should be as specific and concise as possible. "
            "Only provide the translation in natural language (not in LaTeX). "
            "Do not include the original formula in your response. Your response should be less than 30 words. "
            "Write a variable as its name, with no dollar signs. "
            "Do not provide any additional information, explanations, or context."
        )
    },
    {
        "role": "user",
        "content": ""
    },
]


def _strict_eqn():
    """Return PyDetex's ``strict_eqn`` pipeline.

    That pipeline keeps the text of each formula. The ``strict`` pipeline
    replaces a formula with an ``EQUATION_n`` label instead.

    PyDetex imports a Tk button while loading. The text pipeline does not use
    that widget, so a Python install without Tk still runs.
    """
    cached = getattr(_strict_eqn, "_fn", None)
    if cached is not None:
        return cached

    try:
        from pydetex.pipelines import strict_eqn
    except ModuleNotFoundError as exc:
        if exc.name not in {"_tkinter", "tkinter", "tkmacosx"}:
            raise
        stub = types.ModuleType("tkmacosx")
        stub.Button = object
        sys.modules["tkmacosx"] = stub
        for name in list(sys.modules):
            if name == "pydetex" or name.startswith("pydetex."):
                del sys.modules[name]
        from pydetex.pipelines import strict_eqn

    _strict_eqn._fn = strict_eqn
    return strict_eqn


def with_pydetex(text: str | None) -> str:
    """Run PyDetex over ``text``. Text PyDetex cannot parse is left unchanged."""
    if not text:
        return ""
    try:
        return _strict_eqn()(text, show_progress=False)
    except Exception:
        return text


@lru_cache(maxsize=200_000)
def _detex_source(source: str) -> str | None:
    try:
        out = _strict_eqn()(source, show_progress=False)
    except Exception:
        return None
    return _WHITESPACE.sub(" ", out).strip() or None


def pydetex_formula(formula: Formula) -> str | None:
    """Translate one formula with PyDetex, or None if PyDetex fails."""
    return _detex_source(formula.source())


def clean_translation(text: str) -> str:
    """Remove leftover LaTeX delimiters from a model translation.

    Phrases that still contain ``$`` or a LaTeX command are passed through
    pydetex first. Remaining ``$`` signs are then dropped.
    """

    if not text:
        return ""
    cleaned = text
    if "$" in cleaned or "\\" in cleaned:
        cleaned = with_pydetex(cleaned)
    cleaned = cleaned.replace("$", "")
    return _WHITESPACE.sub(" ", cleaned).strip()


def iter_formula_latex():
    """Yield the LaTeX of every formula in the topics, then in the answers."""

    for year in TOPIC_YEARS:
        for _, title, question in iter_clean_topics(year):
            for formula in formulas_in(title) + formulas_in(question):
                yield formula.latex
    for _, body in iter_clean_answers():
        for formula in formulas_in(body):
            yield formula.latex


def _write(path: Path, results: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Translate LaTeX formulas in answers and topics into English.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file to save results.")
    args = parser.parse_args()

    output_file = args.output_file
    filepath = Path(output_file)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    from src.models.gemma import generate, load_model

    model, assistant, processor = load_model()
    results = {}
    if filepath.exists() and filepath.stat().st_size > 0:
        with open(filepath, encoding="utf-8") as file:
            results = json.load(file)
    tried = set(results)
    messages = MESSAGES

    try:
        for latex in tqdm(iter_formula_latex()):
            if latex in tried:
                continue
            tried.add(latex)
            if len(latex) > TRUNCATE_LENGTH:
                continue
            messages[1]["content"] = f"Translate this LaTeX formula to English: {latex}"
            raw = generate(
                messages=messages,
                model=model,
                processor=processor,
                assistant=assistant,
                max_new_tokens=64
            )
            translated = clean_translation(raw)
            if translated:
                results[latex] = translated
        _write(filepath, results)
        print(f"Results saved to {output_file}.")
    except (Exception, KeyboardInterrupt) as e:
        print(f"An error occurred: {e}")
        _write(filepath, results)
        print(f"Partial results saved to {output_file}.")


if __name__ == "__main__":
    main()
