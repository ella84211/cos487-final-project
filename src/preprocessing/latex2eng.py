"""
Translate the LaTeX formulas in answers and topics into English, by PyDetex or by Gemma.

The index script imports this module and reads the translated text directly:

    from src.preprocessing import latex2eng

    translation = latex2eng.by_llm()
    for answer_id, text in translation.answers():
        ...
    for topic_id, title, question in translation.topics(2020):
        ...

Each formula is translated the first time it appears, and the result is saved to a cache file
under data/latex2eng/. Later runs read the cache and only translate formulas that are not in it.
A formula with no usable translation is put back as its original LaTeX.
Run src.preprocessing.clean first; formulas are read from data/clean/.
"""

import re
import sys
import types
from functools import lru_cache
from pathlib import Path

from src.utils.translation import Translation


CACHE_DIR = Path("data/latex2eng")
# Formulas longer than this many characters are not sent to the model.
TRUNCATE_LENGTH = 1000
_WHITESPACE = re.compile(r"\s+")

SYSTEM_PROMPT = (
    "You are a skilled mathematician. Translate the LaTeX formula into natural language. "
    "For example, if the formula is x^2 + y^2 = z^2, the translation is 'x squared plus y squared equals z squared'. "
    "The result will be used in bag-of-words retrieval, so the translation should be as specific and concise as possible. "
    "Only provide the translation in natural language (not in LaTeX). "
    "Do not include the original formula in your response. Your response should be less than 30 words. "
    "Write a variable as its name, with no dollar signs. "
    "Do not provide any additional information, explanations, or context."
)


def by_detex(cache_path: str | Path = CACHE_DIR / "detex.jsonl") -> Translation:
    """Translate each formula with PyDetex."""
    return Translation(_detex_latex, cache_path)


def by_llm(cache_path: str | Path = CACHE_DIR / "llm.jsonl", **model_kwargs) -> Translation:
    """Translate each formula with Gemma.

    The model loads the first time a formula is missing from the cache, so a
    run over a complete cache never loads it. ``model_kwargs`` go to
    ``src.models.gemma.load_model``.
    """
    loaded = {}

    def translate(latex: str) -> str | None:
        if len(latex) > TRUNCATE_LENGTH:
            return None
        if not loaded:
            from src.models.gemma import generate, load_model

            model, assistant, processor = load_model(**model_kwargs)
            loaded.update(generate=generate, model=model, assistant=assistant, processor=processor)
        raw = loaded["generate"](
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate this LaTeX formula to English: {latex}"},
            ],
            model=loaded["model"],
            processor=loaded["processor"],
            assistant=loaded["assistant"],
            max_new_tokens=64,
        )
        return clean_translation(raw) or None

    return Translation(translate, cache_path)


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


def with_pydetex(text: str | None) -> str:
    """Run PyDetex over ``text``. Text PyDetex cannot parse is left unchanged."""
    if not text:
        return ""
    try:
        return _strict_eqn()(text, show_progress=False)
    except Exception:
        return text


def _detex_latex(latex: str) -> str | None:
    return _detex_source(f"${latex}$")


@lru_cache(maxsize=200_000)
def _detex_source(source: str) -> str | None:
    try:
        out = _strict_eqn()(source, show_progress=False)
    except Exception:
        return None
    return _WHITESPACE.sub(" ", out).strip() or None


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
