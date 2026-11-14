# Utils

`src/utils.py` works with the cleaned data written by `python -m src.preprocessing.clean`.

Each row of a file in `data/clean/` stores its fields as a list of pieces. Text is a `str` and a formula is a dict:

```python
["Let ", {"latex": "x^2"}, " be even."]
```

```python
from src.utils import get_formulas, get_text, load, splice

latex = [formula["latex"] for formula in get_formulas(pieces)]  # the formulas to translate
text = get_text(pieces)                                         # the text around them
original = splice(pieces)                                       # text with the original $...$ formulas
rows = load("data/translations/by_detex.json")
```

### `get_formulas(pieces)`

Returns the formula pieces, in order.

### `get_text(pieces)`

Returns the text pieces, in order, with no formulas.

### `splice(pieces, translate=None)`

Joins pieces into text. Each formula is replaced by `translate(formula)`, or kept as `$latex$` when that returns None or there is no `translate`.

### `load(translations)`

Drops the formulas from each row of `data/clean/answers.jsonl` and appends the saved translation for that id. `translations` is a JSON object of `{id: text}`.
