# Retrieval

## index

Creates inverted indexes and contains functions for loading and using them. Run from the repository root to create the indexes. Examples:

```bash
python -m src.retrieval.index.py
python -m src.retrieval.index.py --index_dir data/indexes --approaches with_latex
python -m src.retrieval.index.py --force --index_dir data/indexes --approaches by_detex without_context with_latex
```

### `load_index(preprocessing)`

Loads the inverted index for the specified preprocessing approach. Returns a dictionary representing the index.

### `get_term_frequency(index, term)`

Returns the term frequency for the specified term across all documents in the index.

### `get_document_length(index, docid)`

Returns the length of the specified document, or None if the document is not found.

### `get_num_documents(index)`

Returns the total number of documents in the index.

### `get_average_document_length(index)`

Returns the average document length.

## tfidf

Retrieves documents for a given query, for a specified preprocessing approach, with the TF-IDF ranking function. Contains the `retrieve()` function, which returns a dictionary of the top-k retrieval results, sorted descending by score. Example usage:

```python
from src.retrieval.tfidf import retrieve
results = retrieve("what is tf-idf", "with_latex", 10)
```

## bm25

Retrieves documents for a given query, for a specified preprocessing approach, with the BM25 ranking function. Contains the `retrieve()` function, which returns a dictionary of the top-k retrieval results, sorted descending by score. Example usage:

```python
from src.retrieval.bm25 import retrieve
results = retrieve("what is bm25", "with_latex", 10)
```

## all-mpnet

Retrieves documents for a given query using embeddings from the all-mpnet model and cosine similarity.
FINISH THIS
