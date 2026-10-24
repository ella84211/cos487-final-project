# Utils

Shared code for building a preprocessing approach.

## translation

Wraps a function that translates one LaTeX formula. Every `latex2eng` function returns a `Translation`.

```python
from src.utils.translation import Translation

def translate(latex):
    return ...  # English text, or None to keep the original LaTeX

translation = Translation(translate, "data/my_approach/cache.jsonl")

for answer_id, text in translation.answers():
    ...

for topic_id, title, question in translation.topics(2020):
    ...
```

### `Translation(translate_one, cache_path)`

Calls `translate_one(latex)` once per distinct formula and caches the result in `cache_path`.

### `.answers()`

Yields `(answer_id, text)` for every answer, with each formula replaced by its translation.

### `.topics(year)`

Yields `(topic_id, title, question)` for the topics of `year` (2020, 2021, or 2022).

## formulas

Splits text into plain strings and formulas.

```python
from src.utils.formulas import split_formulas, splice

pieces = split_formulas(html)
text = splice(pieces, lambda formula: ...)
```

### `split_formulas(html)`

Returns a list of `str` and `Formula` pieces. A `Formula` has `latex`, `display` (True for `$$...$$`), and `position`.

### `splice(pieces, translate=None)`

Joins pieces back into text. Each formula is replaced by `translate(formula)`, or kept as its original LaTeX when that returns None.

## arqmath

Reads the cleaned data written by `python -m src.preprocessing.clean`.

### `iter_clean_answers()`

Yields `(answer_id, pieces)` for every answer.

### `iter_clean_topics(year)`

Yields `(topic_id, title_pieces, question_pieces)` for the topics of `year`.
