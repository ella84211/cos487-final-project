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

Translates each LaTeX formula into English and writes `{id: english}` JSON. Indexing appends that file with `load`.

```bash
python -m src.preprocessing.latex2eng                 # PyDetex, answers and topics
python -m src.preprocessing.latex2eng --llm           # Gemma instead of PyDetex
python -m src.preprocessing.latex2eng --detex --llm   # both
python -m src.preprocessing.latex2eng --answers       # answers only
```

Answer files are `data/translations/by_detex.json` and `data/translations/by_llm.json`. Topic files are `data/translations/by_detex_topics.json` and `data/translations/by_llm_topics.json`.

```python
from src.preprocessing import latex2eng

latex2eng.by_detex("data/clean/answers.jsonl")   # writes by_detex.json, returns {id: english}
half = latex2eng.by_detex(r"\frac{1}{2}")        # "½", writes nothing
```

### `by_detex(source, out=...)`

Translates with PyDetex. A cleaned `.jsonl` file, or a list of them, is saved to `out` as `{id: english}` and returned. A string is translated and not saved. A formula with no translation is stored as `$latex$`.

### `by_llm(source, out=...)`

Same as `by_detex`, with Gemma. The model loads on the first call and stays loaded.

## latex2sem

Generates natural language semantic descriptions of formulas using Gemma and saves `{id: description}` JSON. Indexing appends those files with `load`.

```bash
python -m src.preprocessing.latex2sem
python -m src.preprocessing.latex2sem --context
```

The descriptions are saved to `data/semantic_descriptions/` as `semantic_desc_no_context.json`, `semantic_desc_no_context_topics.json`, `semantic_desc_with_context.json`, and `semantic_desc_with_context_topics.json`. For further arguments, see `latex2sem.py`.

## baseline_approaches

### `with_latex(source)`

Given a cleaned `.jsonl` file, returns `{id: text}` with the original formulas kept.

### `without_latex(source)`

Given a cleaned `.jsonl` file, returns `{id: text}` with the formulas removed.
