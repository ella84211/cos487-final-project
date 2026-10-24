# Preprocessing

Run `python -m src.preprocessing.clean` once before indexing. It writes the cleaned answers and topics to `data/clean/`.

## latex2eng

Translates each LaTeX formula into English and splices it back into the text.

```python
from src.preprocessing import latex2eng

translation = latex2eng.by_detex()  # or latex2eng.by_llm()

for answer_id, text in translation.answers():
    ...

for topic_id, title, question in translation.topics(2020):
    ...
```

### `by_detex(cache_path="data/latex2eng/detex.jsonl")`

Translates formulas with PyDetex.

### `by_llm(cache_path="data/latex2eng/llm.jsonl", **model_kwargs)`

Translates formulas with Gemma. `model_kwargs` are passed to `src.models.gemma.load_model`. The model loads only when a formula is missing from the cache.

Both functions cache each translation as soon as it's made, so a rerun only translates formulas that aren't in the cache yet. A formula with no translation is kept as its original LaTeX.
