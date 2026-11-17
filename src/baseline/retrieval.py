"""
Retrieves top-k results by cosine similarity using a pretrained or finetuned model. 
Contains the retrieval(query, training, k) function, which retrieves the top-k similar documents for a given query.
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Literal

from src.utils import INDEX_MAP, MODEL_DIR

PATHS = {
    "pretrained": {
        "model": "all-mpnet-base-v2",
        "embeddings": INDEX_MAP["neural_pt"],
    },
    "finetuned": {
        "model": str(MODEL_DIR / "finetuned_model"),
        "embeddings": INDEX_MAP["neural_ft"],
    },
}

models = {}
embeddings_files = {}


def retrieval(query: str, training: Literal["pretrained", "finetuned"], k=1000):
    """Retrieves top-k results by cosine similarity using the model with the given training option"""
    if training not in PATHS:
        raise ValueError(f"Invalid training option: {training}. Must be one of {list(PATHS.keys())}.")
    global models
    if PATHS[training]["model"] in models:
        model = models[PATHS[training]["model"]]
    else:
        model = SentenceTransformer(PATHS[training]["model"])
        models[PATHS[training]["model"]] = model
    answers = _get_embeddings(PATHS[training]["embeddings"])
    answer_ids = answers["ids"]
    answer_embeddings = answers["embeddings"]
    query_embedding = model.encode(query, convert_to_numpy=True)
    results = {}
    hits = util.semantic_search(query_embedding, answer_embeddings, top_k=k)
    hits = hits[0]
    for hit in hits:
        index = hit['corpus_id']
        answer_id = answer_ids[index]
        score = hit['score']
        results[answer_id] = score
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    return results


def _get_embeddings(filepath):
    """Loads embeddings from embeddings file"""
    global embeddings_files
    if filepath in embeddings_files:
        return embeddings_files[filepath]
    data = np.load(filepath)
    answer_ids = [str(id) for id in list(data["ids"])]
    embeddings = data["embeddings"]
    answers = {"ids": answer_ids, "embeddings": embeddings}
    embeddings_files[filepath] = answers
    return answers
