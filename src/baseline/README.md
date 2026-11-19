# Retrieval

## finetune

Fine-tunes the all-mpnet-base-v2 model on ARQMath-1 and -2. Writes to models/finetuned_model. Run from the repository root.

Usage:

```bash
python -m src.baseline.finetune
```

## index

Uses the pretrained and fine-tuned model to encode the cleaned answers in ``data/clean/answers.jsonl`` and saves the embeddings to .npz files. Run from the repository root.

Usage:

```bash
python -m src.baseline.index
```

## retrieval

Retrieves the top-k similar documents for a given query using the pretrained or fine-tuned model. The training status of the model can be specified as "pretrained" or "finetuned".
Usage:

```python
from src.baseline.retrieval import retrieval

results = retrieval(query="What is the pythagorean theorem?", training="pretrained", k=10)
```
