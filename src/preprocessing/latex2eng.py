"""
Translate the LaTeX formulas in answers and topics into English, by PyDetex or by Gemma.

Each run writes ``{id: english}`` JSON. The English is the translations of that
document's formulas, in order. Indexing appends the file with ``src.utils.load``.
A formula with no usable translation is stored as its original ``$latex$``.
Given a single string, each function returns that string's translation and writes nothing.
A Gemma run keeps document ids already in its output file and continues after them.
``--force_restart`` translates those documents again.

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

from src.utils import CLEANED_ANSWERS, CLEANED_TOPICS, TRANSLATION_DIR, get_formulas


# Formulas longer than this many characters are not sent to the model.
TRUNCATE_LENGTH = 1000

TRANSLATIONS_DETEX = TRANSLATION_DIR / "by_detex.json"
TRANSLATIONS_LLM = TRANSLATION_DIR / "by_llm.json"
TRANSLATIONS_DETEX_TOPICS = TRANSLATION_DIR / "by_detex_topics.json"
TRANSLATIONS_LLM_TOPICS = TRANSLATION_DIR / "by_llm_topics.json"

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


def by_llm(source: str | Path | list, out: str | Path = TRANSLATIONS_LLM, batch_size: int = 8, force_restart: bool = False) -> dict | str:
    """Translate with Gemma and, for a cleaned file, write ``{id: english}``.

    Given a cleaned ``.jsonl`` file or a list of them, writes ``out`` and returns that dictionary.
    A document id already in ``out`` is kept and not translated again.
    With ``force_restart``, every document is translated again.
    Given one LaTeX formula, returns its translation, or the formula unchanged
    when there is none. The model loads on the first call and stays loaded.
    """
    if isinstance(source, list) or _is_file(source):
        return _save(_llm, source, Path(out), translate_batch=_llm_many, batch_size=batch_size, force_restart=force_restart)
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
    return _finish_llm(reply)


def _llm_many(formulas: list[str], batch_size: int) -> list[str | None]:
    """Translate distinct formulas in one batched Gemma call."""
    from src.preprocessing.gemma import generate

    model, _, processor = _gemma()
    indexes = []
    conversations = []
    results: list[str | None] = [None] * len(formulas)
    for i, latex in enumerate(formulas):
        if len(latex) > TRUNCATE_LENGTH:
            continue
        indexes.append(i)
        conversations.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate this LaTeX formula to English: {latex}"},
        ])
    if not conversations:
        return results
    replies = generate(
        messages=conversations,
        model=model,
        processor=processor,
        max_new_tokens=64,
        batch_size=batch_size,
    )
    for i, reply in zip(indexes, replies):
        results[i] = _finish_llm(reply)
    return results


def _finish_llm(reply: str | None) -> str | None:
    if not reply:
        return None
    if "\\" in reply:
        reply = _detex(reply) or reply
    return reply.replace("$", "")


def _save(translate, source: str | Path | list, out: Path, translate_batch=None, batch_size: int = 1, force_restart: bool = False) -> dict:
    """Translate every formula in each cleaned row and write ``{id: english}``."""
    if translate_batch is not None:
        return _save_batched(translate_batch, source, out, batch_size, force_restart=force_restart)
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


def _save_batched(translate_batch, source: str | Path | list, out: Path, batch_size: int, force_restart: bool = False) -> dict:
    """Like ``_save``, but translates new formulas in groups of ``batch_size``.

    Document ids already stored in ``out`` are skipped. With ``force_restart``,
    those ids are translated again. The file is written when the run finishes,
    and when it stops from an error or interrupt.
    """
    cache = {}
    results = {} if force_restart else _read(out)
    resumed = len(results)
    if force_restart:
        print(f"Restarting translations for {out}.")
    elif resumed:
        print(f"Resuming from {resumed} translations in {out}.")
    held = []
    pending = []
    pending_seen = set()

    def render(latexes):
        parts = []
        for latex in latexes:
            english = cache.get(latex)
            parts.append(english.strip() if english and english.strip() else f"${latex}$")
        return " ".join(parts)

    def flush():
        if pending:
            translated = translate_batch(pending, batch_size)
            if len(translated) != len(pending):
                raise RuntimeError(f"Expected {len(pending)} translations, got {len(translated)}.")
            for latex, english in zip(pending, translated):
                cache[latex] = english
            pending.clear()
            pending_seen.clear()
        for doc_id, latexes in held:
            results[doc_id] = render(latexes)
        held.clear()

    try:
        for path in source if isinstance(source, list) else [source]:
            with open(path, encoding="utf-8") as file:
                for line in tqdm(file, desc=Path(path).name):
                    row = json.loads(line)
                    doc_id = str(row.pop("id"))
                    if doc_id in results:
                        continue
                    pieces = [piece for field in row.values() if isinstance(field, list) for piece in field]
                    formulas = get_formulas(pieces)
                    if not formulas:
                        continue
                    latexes = [formula["latex"] for formula in formulas]
                    for latex in latexes:
                        if latex in cache or latex in pending_seen:
                            continue
                        pending.append(latex)
                        pending_seen.add(latex)
                    if pending:
                        held.append((doc_id, latexes))
                        if len(pending) >= batch_size:
                            flush()
                    else:
                        results[doc_id] = render(latexes)
        flush()
        _write(results, out)
        new = len(results) - resumed
        print(f"Results saved to {out}. {new} new, {resumed} already translated.")
        return results
    except (Exception, KeyboardInterrupt):
        _write(results, out)
        print(f"Partial results saved to {out}.")
        raise


def _read(out: Path) -> dict:
    """Return the ``{id: english}`` object already stored in ``out``, or an empty dict."""
    if not out.exists():
        return {}
    with open(out, encoding="utf-8") as file:
        saved = json.load(file)
    if not isinstance(saved, dict):
        raise ValueError(f"Expected a JSON object of translations in {out}.")
    return {str(key): value for key, value in saved.items()}


def _write(results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    tmp.replace(out)


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
    parser.add_argument("--batch_size", type=int, default=8, help="Formulas per Gemma forward pass. Used with --llm.")
    parser.add_argument("--force_restart", action="store_true", help="Ignore saved Gemma translations and start again. Used with --llm.")
    args = parser.parse_args()

    run_detex = args.detex or not args.llm
    run_llm = args.llm
    if args.force_restart and not run_llm:
        parser.error("--force_restart requires --llm")
    do_answers = args.answers or not args.topics
    do_topics = args.topics or not args.answers

    if run_detex and do_answers:
        by_detex(CLEANED_ANSWERS, TRANSLATIONS_DETEX)
    if run_detex and do_topics:
        by_detex(CLEANED_TOPICS, TRANSLATIONS_DETEX_TOPICS)
    if run_llm and do_answers:
        by_llm(CLEANED_ANSWERS, TRANSLATIONS_LLM, batch_size=args.batch_size, force_restart=args.force_restart)
    if run_llm and do_topics:
        by_llm(CLEANED_TOPICS, TRANSLATIONS_LLM_TOPICS, batch_size=args.batch_size, force_restart=args.force_restart)


if __name__ == "__main__":
    main()
