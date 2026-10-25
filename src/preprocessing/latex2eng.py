"""
Translate the LaTeX formulas in answers and topics into English, by PyDetex or by Gemma.

Given one cleaned file from data/clean/, each function returns {id: text} for every row in it,
with every formula replaced by English. A topic's text is its title and question joined.
A formula with no usable translation is kept as its original LaTeX.
Run src.preprocessing.clean first to write the cleaned files.
Given a single string, each function returns that string's translation.

Example usage:
    from src.preprocessing import latex2eng

    answers = latex2eng.by_llm("data/clean/answers.jsonl")
    topics = latex2eng.by_llm("data/clean/topics_2020.jsonl")
    half = latex2eng.by_detex(r"\\frac{1}{2}")
"""

import json
from functools import cache
from pathlib import Path

from pydetex.pipelines import strict_eqn
from tqdm import tqdm

from src.utils import splice


# Formulas longer than this many characters are not sent to the model.
TRUNCATE_LENGTH = 1000

SYSTEM_PROMPT = (
    "You are a skilled mathematician. Translate the LaTeX formula into natural language. "
    "For example, if the formula is x^2 + y^2 = z^2, the translation is 'x squared plus y squared equals z squared'. "
    "The result will be used in bag-of-words retrieval, so the translation should be as specific and concise as possible. "
    "Only provide the translation in natural language (not in LaTeX). "
    "Do not include the original formula in your response. Your response should be less than 30 words. "
    "Write a variable as its name, with no dollar signs. "
    "Do not provide any additional information, explanations, or context."
)


def by_detex(source: str | Path) -> dict | str:
    """Translate with PyDetex.

    Given a cleaned ``.jsonl`` file, returns ``{id: text}`` for every row in it.
    Given a string, returns its translation: a bare formula like ``\\frac{1}{2}``,
    or text with ``$``-delimited formulas like an LLM reply. A string PyDetex
    cannot translate is returned unchanged.
    """
    if _is_file(source):
        return _translate(lambda latex: _detex(f"${latex}$"), source)
    return _detex(source if "$" in source else f"${source}$") or source


def by_llm(source: str | Path) -> dict | str:
    """Translate with Gemma.

    Given a cleaned ``.jsonl`` file, returns ``{id: text}`` for every row in it.
    Given one LaTeX formula, returns its translation, or the formula unchanged
    when there is none. The model loads on the first call and stays loaded.
    """
    if _is_file(source):
        return _translate(_llm, source)
    return _llm(source) or source


def _is_file(source: str | Path) -> bool:
    return isinstance(source, Path) or source.endswith(".jsonl")


@cache
def _gemma():
    from src.models.gemma import load_model

    return load_model()


def _llm(latex: str) -> str | None:
    if len(latex) > TRUNCATE_LENGTH:
        return None
    from src.models.gemma import generate

    model, assistant, processor = _gemma()
    reply = generate(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate this LaTeX formula to English: {latex}"},
        ],
        model=model,
        processor=processor,
        assistant=assistant,
        max_new_tokens=64,
    )
    if "\\" in reply:
        reply = _detex(reply) or reply
    return reply.replace("$", "")


def _translate(translate, path: str | Path) -> dict:
    """Splice each row's formulas through ``translate``, calling it once per distinct formula."""
    translate = cache(translate)

    def replace(formula):
        return translate(formula["latex"])

    texts = {}
    with open(path, encoding="utf-8") as file:
        for line in tqdm(file, desc=Path(path).name):
            row = json.loads(line)
            row_id = row.pop("id")
            texts[row_id] = " ".join(splice(field, replace) for field in row.values()).strip()
    return texts


def _detex(text: str) -> str | None:
    try:
        return " ".join(strict_eqn(text, show_progress=False).split()) or None
    except Exception:
        return None
