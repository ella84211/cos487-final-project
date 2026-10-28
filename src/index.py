"""
Creates inverted indexes for the selected preprocessing approaches. 
Writes to the directory specified by the --index_dir flag, or to data/indexes/ by default. 
Force rebuild can be enabled with the --force flag. 
Preprocessing approaches can be selected with the --approaches flag, or all six will be built by default. 
The cleaned data file must be specified with the --cleaned_data flag. The default cleaned data file is data/clean/answers.jsonl. 

For with_context and without_context, the semantic description files must be specified with the --semantic_desc_with_context and 
--semantic_desc_no_context flags. They default to data/semantic_descriptions/semantic_desc_with_context.json and 
data/semantic_descriptions/semantic_desc_no_context.json.

Example usage:
    python -m src.index --force --index_dir data/indexes --approaches by_detex without_context with_latex

The following functions load and use the indexes:
    load_index(preprocessing: str) -> dict                          # Loads the inverted index for the specified preprocessing approach.
    get_term_frequency(index: dict, term: str) -> dict              # Returns the term frequency for the specified term across all documents.
    get_document_length(index: dict, docid: str) -> int | None      # Returns the length of the specified document, or None if the document is not found.
    get_num_documents(index: dict) -> int                           # Returns the total number of documents in the index.
    get_average_document_length(index: dict) -> float               # Returns the average document length in the index.
"""


import json
import os
import argparse
import time
import nltk

from pathlib import Path
from tqdm import tqdm
from nltk.corpus import stopwords

from src.preprocessing.latex2eng import by_llm, by_detex
from src.preprocessing.latex2sem import with_context, without_context
from src.preprocessing.baseline_approaches import with_latex, without_latex


INDEX_DIR = os.path.join("data", "indexes")

PREPROCESSING_OPTIONS = {
    "by_detex": by_detex,
    "by_llm": by_llm,
    "without_context": without_context,
    "with_context": with_context,
    "with_latex": with_latex,
    "without_latex": without_latex,
}


nltk_data_dir = Path.home() / ".cache" / "cos487-nltk"
nltk_data_dir.mkdir(parents=True, exist_ok=True)
nltk_data_dir.chmod(0o700)

nltk.data.path.insert(0, str(nltk_data_dir))

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    success = nltk.download(
        "stopwords",
        download_dir=str(nltk_data_dir),
        quiet=True,
        raise_on_error=True,
    )
    if not success:
        raise RuntimeError("Could not download NLTK stopwords")

stop_words = set(stopwords.words("english"))
punctuation = ".,!?;:()[]{}\"'"


def split_words(text: str) -> list[str]:
    """Split a string into words, removing punctuation and whitespace."""

    return [
        word.strip(punctuation)
        for word in text.lower().split()
        if word.strip(punctuation)
        and word.strip(punctuation) not in stop_words
    ]


def build_index(preprocessing: str, source: str | Path, semantic_desc: str | Path = None, force_rebuild: bool=False):
    """Builds an inverted index for the given preprocessing option."""
    if preprocessing not in PREPROCESSING_OPTIONS:
        raise ValueError(f"Invalid preprocessing option: {preprocessing}. Must be one of {list(PREPROCESSING_OPTIONS.keys())}")
    index_path = f"{INDEX_DIR}/{preprocessing}_index.json"
    if os.path.exists(index_path) and not force_rebuild:
        print(f"Index file {index_path} already exists. Skipping index build.")
        return
    
    start = time.time()
    document_lengths = {}
    terms = {}
    index = {
        "doc_lens": document_lengths,
        "terms": terms,
    }
    data = PREPROCESSING_OPTIONS[preprocessing](source) if semantic_desc is None else PREPROCESSING_OPTIONS[preprocessing](source, semantic_desc)
    for docid, text in tqdm(data.items(), desc=f"Building {preprocessing} index"):
        words = split_words(text)
        document_lengths[docid] = len(words)
        for word in words:
            if word not in terms:
                terms[word] = {}
            if docid not in terms[word]:
                terms[word][docid] = 0
            terms[word][docid] += 1

    num_docs = len(document_lengths)
    avg_doc_len = sum(document_lengths.values()) / num_docs if num_docs > 0 else 0.0
    index["num_docs"] = num_docs
    index["avg_doc_len"] = avg_doc_len

    with open(index_path, "w", encoding="utf-8") as file:
        json.dump(index, file, indent=4)
    print(f"Built {preprocessing} index, saved to {index_path}")
    print(f"Time taken: {(time.time() - start) / 60:.2f} minutes\n")


def load_index(preprocessing: str) -> dict:
    """Loads the inverted index for the given preprocessing option from the INDEX_DIR directory."""
    if preprocessing not in PREPROCESSING_OPTIONS:
        raise ValueError(f"Invalid preprocessing option: {preprocessing}. Must be one of {list(PREPROCESSING_OPTIONS.keys())}")
    index_path = f"{INDEX_DIR}/{preprocessing}_index.json"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file {index_path} does not exist. Please build the index first.")
    
    print(f"Loading index for preprocessing option '{preprocessing}'...")
    with open(index_path, "r", encoding="utf-8") as file:
        index = json.load(file)
    return index


def get_term_frequency(index: dict, term: str) -> dict:
    """Gets the term frequency for a given term from the inverted index"""
    if term not in index["terms"]:
        return {}
    return index["terms"][term]


def get_document_length(index: dict, docid: str) -> int | None:
    """Gets the length of a document in the inverted index, or None if the document is not found."""
    return index["doc_lens"].get(docid, None)


def get_num_documents(index: dict) -> int:
    """Gets the number of documents in the given inverted index"""
    return index["num_docs"]


def get_average_document_length(index: dict) -> float:
    """Gets the average document length in the given inverted index"""
    return index["avg_doc_len"]


def main():
    """Main function to build the indexes for both paragraph and paper levels, with and without stemming."""
    approaches = list(PREPROCESSING_OPTIONS.keys())
    parser = argparse.ArgumentParser(description="Build inverted indexes for the corpus.")
    parser.add_argument("--force", action="store_true", help="Force rebuild of indexes even if they already exist.")
    parser.add_argument("--index_dir", type=str, help="Directory to save the indexes.")
    parser.add_argument("--cleaned_data", type=str, default="data/clean/answers.jsonl", help="Path to the cleaned JSONL file containing answers.")
    parser.add_argument("--semantic_desc_with_context", type=str, 
                        default="data/semantic_descriptions/semantic_desc_with_context.json", 
                        help="Path to the JSON file containing semantic descriptions with context.")
    parser.add_argument("--semantic_desc_no_context", type=str, 
                        default="data/semantic_descriptions/semantic_desc_no_context.json", 
                        help="Path to the JSON file containing semantic descriptions without context.")

    parser.add_argument(
        "--approaches",
        nargs="+",
        choices=approaches,
        default=approaches,
        help=(
            "Models to run. Choose one or more: "
            f"{', '.join(approaches)}. "
            "Default: all six."
        ),
    )

    args = parser.parse_args()
    if args.index_dir:
        global INDEX_DIR
        INDEX_DIR = args.index_dir
    os.makedirs(INDEX_DIR, exist_ok=True)
    for preprocessing in args.approaches:
        if preprocessing in ["with_context", "without_context"]:
            sem_desc = args.semantic_desc_with_context if preprocessing == "with_context" else args.semantic_desc_no_context
            build_index(preprocessing, source=args.cleaned_data, semantic_desc=sem_desc, force_rebuild=args.force)
        else:
            build_index(preprocessing, source=args.cleaned_data, force_rebuild=args.force)


if __name__ == "__main__":
    main()
