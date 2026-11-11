"""
Translate the LaTeX formulas in answers and topics into English, by PyDetex or by Gemma.

Each run writes ``{id: english}`` JSON. The English is the translations of that
document's formulas, in order. Indexing appends the file with ``src.utils.load``.
A formula with no usable translation is stored as its original ``$latex$``.
Given a single string, each function returns that string's translation and writes nothing.

Example usage:
    python -m src.preprocessing.latex2eng
    python -m src.preprocessing.latex2eng --llm --answers

    from src.preprocessing import latex2eng
    latex2eng.by_detex("data/clean/answers.jsonl")
    half = latex2eng.by_detex(r"\\frac{1}{2}")
"""

import argparse
import json
from functools import cache
from pathlib import Path

from pydetex.pipelines import strict_eqn
from tqdm import tqdm

from src.utils import get_formulas


# Formulas longer than this many characters are not sent to the model.
TRUNCATE_LENGTH = 1000

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CLEANED_ANSWERS = DATA_DIR / "clean" / "answers.jsonl"
CLEANED_TOPICS = [
    DATA_DIR / "clean" / "topics_2020.jsonl",
    DATA_DIR / "clean" / "topics_2021.jsonl",
    DATA_DIR / "clean" / "topics_2022.jsonl",
]

TRANSLATIONS_DETEX = DATA_DIR / "translations" / "by_detex.json"
TRANSLATIONS_LLM = DATA_DIR / "translations" / "by_llm.json"
TRANSLATIONS_DETEX_TOPICS = DATA_DIR / "translations" / "by_detex_topics.json"
TRANSLATIONS_LLM_TOPICS = DATA_DIR / "translations" / "by_llm_topics.json"

SYSTEM_PROMPT = (
    "You are a skilled mathematician. Translate the LaTeX formula into natural language. "
    "For example, if the formula is x^2 + y^2 = z^2, the translation is 'x squared plus y squared equals z squared'. "
    "The result will be used in bag-of-words retrieval, so the translation should be as specific and concise as possible. "
    "Only provide the translation in natural language (not in LaTeX). "
    "Do not include the original formula in your response. Your response should be less than 30 words. "
    "Write a variable as its name, with no dollar signs. "
    "Do not provide any additional information, explanations, or context."
)


def by_detex(source: str | Path | list, out: str | Path = TRANSLATIONS_DETEX) -> dict | str:
    """Translate with PyDetex and, for a cleaned file, write ``{id: english}``.

    Given a cleaned ``.jsonl`` file or a list of them, writes ``out`` and returns that dictionary.
    Given a string, returns its translation: a bare formula like ``\\frac{1}{2}``,
    or text with ``$``-delimited formulas like an LLM reply. A string PyDetex
    cannot translate is returned unchanged.
    """
    if isinstance(source, list) or _is_file(source):
        return _save(lambda latex: _detex(f"${latex}$"), source, Path(out))
    return _detex(source if "$" in source else f"${source}$") or source


def by_llm(source: str | Path | list, out: str | Path = TRANSLATIONS_LLM) -> dict | str:
    """Translate with Gemma and, for a cleaned file, write ``{id: english}``.

    Given a cleaned ``.jsonl`` file or a list of them, writes ``out`` and returns that dictionary.
    Given one LaTeX formula, returns its translation, or the formula unchanged
    when there is none. The model loads on the first call and stays loaded.
    """
    if isinstance(source, list) or _is_file(source):
        return _save(_llm, source, Path(out))
    return _llm(source) or source


def _is_file(source: str | Path) -> bool:
    return isinstance(source, Path) or source.endswith(".jsonl")


@cache
def _gemma():
    from src.preprocessing.gemma import load_model

    return load_model()


def _llm(latex: str) -> str | None:
    if len(latex) > TRUNCATE_LENGTH:
        return None
    from src.preprocessing.gemma import generate

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


def _save(translate, source: str | Path | list, out: Path) -> dict:
    """Translate every formula in each cleaned row and write ``{id: english}``."""
    translate = cache(translate)
    results = {}
    try:
        for path in source if isinstance(source, list) else [source]:
            with open(path, encoding="utf-8") as file:
                for line in tqdm(file, desc=Path(path).name):
                    row = json.loads(line)
                    doc_id = str(row.pop("id"))
                    pieces = [piece for field in row.values() if isinstance(field, list) for piece in field]
                    formulas = get_formulas(pieces)
                    if not formulas:
                        continue
                    parts = []
                    for formula in formulas:
                        english = translate(formula["latex"])
                        parts.append(english.strip() if english and english.strip() else f"${formula['latex']}$")
                    results[doc_id] = " ".join(parts)
        _write(results, out)
        print(f"Results saved to {out}.")
        return results
    except (Exception, KeyboardInterrupt):
        _write(results, out)
        print(f"Partial results saved to {out}.")
        raise


def _write(results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def _detex(text: str) -> str | None:
    try:
        return " ".join(strict_eqn(text, show_progress=False).split()) or None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Translate LaTeX formulas to English and save them.")
    parser.add_argument("--llm", action="store_true", help="Translate with Gemma.")
    parser.add_argument("--detex", action="store_true", help="Translate with PyDetex. This is the default.")
    parser.add_argument("--answers", action="store_true", help="Translate answers.")
    parser.add_argument("--topics", action="store_true", help="Translate topics.")


    # parser.add_argument("--cleaned_file", type=str, default=CLEANED_ANSWERS, help="Cleaned answers JSONL file.")
    # parser.add_argument("--cleaned_topics_files", nargs="*", default=CLEANED_TOPICS, help="Cleaned topic JSONL files.")
    args = parser.parse_args()

    run_detex = args.detex or not args.llm
    run_llm = args.llm
    do_answers = args.answers or not args.topics
    do_topics = args.topics or not args.answers

    if run_detex and do_answers:
        by_detex(args.cleaned_file, TRANSLATIONS_DETEX)
    if run_detex and do_topics:
        by_detex(args.cleaned_topics_files, TRANSLATIONS_DETEX_TOPICS)
    if run_llm and do_answers:
        by_llm(args.cleaned_file, TRANSLATIONS_LLM)
    if run_llm and do_topics:
        by_llm(args.cleaned_topics_files, TRANSLATIONS_LLM_TOPICS)


if __name__ == "__main__":
    main()
