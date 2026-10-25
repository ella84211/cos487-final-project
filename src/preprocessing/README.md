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
