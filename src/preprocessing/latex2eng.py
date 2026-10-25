"""
Translate the LaTeX formulas in answers and topics into English, by PyDetex or by Gemma.

Each function reads one cleaned file from data/clean/ and returns {id: text} for every row in it,
with every formula replaced by English. A topic's text is its title and question joined.
A formula with no usable translation is kept as its original LaTeX.
Run src.preprocessing.clean first to write the cleaned files.

Example usage:
    from src.preprocessing import latex2eng

    answers = latex2eng.by_llm("data/clean/answers.jsonl")
    topics = latex2eng.by_llm("data/clean/topics_2020.jsonl")
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


def by_detex(path: str | Path) -> dict:
    """Return ``{id: text}`` for every row of the cleaned file at ``path``, translated by PyDetex."""
    return _translate(lambda latex: _detex(f"${latex}$"), path)


def by_llm(path: str | Path) -> dict:
    """Return ``{id: text}`` for every row of the cleaned file at ``path``, translated by Gemma."""
    from src.models.gemma import generate, load_model

    model, assistant, processor = load_model()

    def translate(latex: str) -> str | None:
        if len(latex) > TRUNCATE_LENGTH:
            return None
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

    return _translate(translate, path)


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
