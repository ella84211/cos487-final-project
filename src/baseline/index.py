"""
Encodes the cleaned answers using a pretrained and finetuned model and saves the embeddings to .npz files.

Usage:
    python -m src.baseline.index
"""

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.utils import CLEANED_ANSWERS, INDEX_MAP, MODEL_DIR, MODEL, load

PT_MODEL = "all-mpnet-base-v2"
FT_MODEL = MODEL_DIR / "finetuned_model"

BATCH_SIZE = 128

def read_corpus():
    """Return ``{answer_id: text}`` for the cleaned answers, with formulas kept."""
    return load(CLEANED_ANSWERS, with_latex=True)


def get_embeddings(model_Path, filepath, candidates):
    """Encodes the answers and saves them to a .npz file if they don't already exist"""
    if Path(filepath).is_file():
        print(f"Embeddings found at {filepath}.")
        return

    model = SentenceTransformer(model_Path)
    answer_ids = list(candidates.keys())
    corpus_texts = list(candidates.values())

    all_embeddings = []
    for i in tqdm(range(0, len(corpus_texts), BATCH_SIZE), desc="Encoding answers"):
        batch = corpus_texts[i:i+BATCH_SIZE]
        emb = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(emb)
    all_embeddings = np.vstack(all_embeddings)
    np.savez_compressed(
        filepath,
        ids=np.array(answer_ids),
        embeddings=all_embeddings
    )
    print(f"Wrote embeddings to {filepath}")


if __name__ == '__main__':
    print("Reading corpus...")
    candidates = read_corpus()
    print(f"Read {len(candidates)} answers from the corpus. Now encoding with the pretrained model...")
    get_embeddings(PT_MODEL, INDEX_MAP["neural_pt"], candidates)
    print(f"Read {len(candidates)} answers from the corpus. Now encoding with the finetuned model...")
    get_embeddings(FT_MODEL, INDEX_MAP["neural_ft"], candidates)
    print("Done.")
