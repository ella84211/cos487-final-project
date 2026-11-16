"""
Encodes all answers in the Posts file using a pretrained and finetuned model and saves the embeddings to .npz files.

Usage:
    python -m src.baseline.index
"""

from sentence_transformers import SentenceTransformer
import bs4
from tqdm import tqdm
import numpy as np
from pathlib import Path
from ARQMathCode.Entity_Parser_Record.post_parser_record import PostParserRecord


THIS_DIR = Path(__file__).resolve().parent

PT_MODEL = "all-mpnet-base-v2"
FT_MODEL = str(THIS_DIR / "finetuned_model")

PT_EMBEDDINGS_FILE = THIS_DIR / "pretrained_answer_embeddings.npz"
FT_EMBEDDINGS_FILE = THIS_DIR / "finetuned_answer_embeddings.npz"

POSTS = THIS_DIR.parent.parent / "data" / "Posts.V1.3.xml"
BATCH_SIZE = 128


def parse_answer(text):
    """Removes HTML tags from answers in the Post file"""
    soup1 = bs4.BeautifulSoup(text, 'html.parser')
    for match in soup1.find_all('span'):
        match.unwrap()
    text = soup1.text
    new_text = text.strip()
    return new_text


def read_corpus():
    """Reads answers from the posts file and returns a dictionary of answer_id: answer_text"""
    post_reader = PostParserRecord(POSTS)
    final_result = {}
    for answer_id, post_obj in tqdm(post_reader.map_just_answers.items(), desc="Parsing answers"):
        body = parse_answer(post_obj.body)
        final_result[answer_id] = body
    return final_result


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
    get_embeddings(PT_MODEL, PT_EMBEDDINGS_FILE, candidates)
    print(f"Read {len(candidates)} answers from the corpus. Now encoding with the finetuned model...")
    get_embeddings(FT_MODEL, FT_EMBEDDINGS_FILE, candidates)
    print("Done.")
