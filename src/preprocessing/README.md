# Preprocessing

## clean

Splits every answer and Task 1 topic into text and formulas, and writes them to `data/clean/`. Run it once before any approach.

```bash
python -m src.preprocessing.clean            # answers and topics
python -m src.preprocessing.clean topics     # only topics
python -m src.preprocessing.clean answers    # only answers
```

It writes `answers.jsonl` with `id` and `body`, and `topics_<year>.jsonl` with `id`, `title`, and `question`. Each field is a list of pieces, described in [`src/README.md`](../README.md). Cleaning the answers takes about 3 minutes and about 7 GB of memory.

## latex2eng

Translates each LaTeX formula into English and splices it back into the text.

```python
from src.preprocessing import latex2eng

answers = latex2eng.by_detex("data/clean/answers.jsonl")     # {answer_id: text}
topics = latex2eng.by_detex("data/clean/topics_2020.jsonl")  # {topic_id: text}
half = latex2eng.by_detex(r"\frac{1}{2}")                    # "½"
```

### `by_detex(source)`

Translates with PyDetex. Given a cleaned `.jsonl` file, returns `{id: text}` for every row in it. A topic's text is its title and question joined. Given a string, returns its translation. The string can be a bare formula, or text with `$`-delimited formulas such as an LLM reply.

### `by_llm(source)`

Translates with Gemma. Given a cleaned `.jsonl` file, returns the same dictionary as `by_detex`. Given one LaTeX formula, returns its translation. The model loads on the first call and stays loaded.

A formula with no translation is kept as its original LaTeX.

## latex2sem

Generates natural language semantic descriptions of formulas using Gemma. Before using `with_context()` or `without_context()`, or the corresponding functions for topics, generate the semantic descriptions by running this file from the repository root:

```bash
python -m src.preprocessing.latex2sem
python -m src.preprocessing.latex2sem --context
```

This generates the semantic descriptions, both with and without the surrounding text context, for both topics and answers. The descriptions are saved to `data/semantic_descriptions/` as `semantic_desc_no_context.json`, `semantic_desc_no_context_topics.json`, `semantic_desc_with_context.json`, and `semantic_desc_with_context_topics.json`. For further arguments, see `latex2sem.py`.

### `with_context(source, semantic_desc)`

Appends the semantic descriptions generated with the surrounding text to each answer's text. Given a cleaned `.jsonl` file and the semantic descriptions, returns the same dictionary as `by_detex`.

### `without_context(source, semantic_desc)`

Appends the semantic descriptions generated without the surrounding text to each answer's text. Given a cleaned `.jsonl` file and the semantic descriptions, returns the same dictionary as `by_detex`.

### `with_context_topics(source, semantic_desc)`

Appends the semantic descriptions generated with the surrounding text to each topic's text. Given a list of cleaned `.jsonl` files and the semantic descriptions (in one file), returns the same dictionary as `by_detex`.

### `without_context_topics(source, semantic_desc)`

Appends the semantic descriptions generated without the surrounding text to each topic's text. Given a list of cleaned `.jsonl` files and the semantic descriptions (in one file), returns the same dictionary as `by_detex`.

## baseline_approaches

### `with_latex(source)`

Given a cleaned `.jsonl` file, returns the same dictionary as `by_detex`, but with the original LaTeX formulas instead of their English translations.

### `without_latex(source)`

Given a cleaned `.jsonl` file, returns the same dictionary as `by_detex`, but with the original LaTeX formulas removed from the text.
