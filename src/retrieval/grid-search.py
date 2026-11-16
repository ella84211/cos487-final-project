"""Tune BM25's k1 and b on ARQMath-1 and ARQMath-2, then compare the runs with ranx.

Uses the 2020 and 2021 Task 1 topics only. The 2022 topics are held out.

    python src/retrieval/grid-search.py --preprocessing with_latex
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ranx import Qrels, Run, compare, evaluate

from src.utils import CLEANED_DIR, DATA_DIR, TRANSLATION_DIR, load


# k1 and b values to try. 1.2 and 0.75 are the BM25 defaults.
K1 = (0.9, 1.2, 1.6)
B = (0.4, 0.75, 1.0)
YEARS = (2020, 2021)

TOPIC_TRANSLATIONS = {
    "by_detex": TRANSLATION_DIR / "by_detex_topics.json",
    "by_llm": TRANSLATION_DIR / "by_llm_topics.json",
    "without_context": TRANSLATION_DIR / "semantic_desc_no_context_topics.json",
    "with_context": TRANSLATION_DIR / "semantic_desc_with_context_topics.json",
}


def main():
    parser = argparse.ArgumentParser(description="Tune BM25 k1 and b on ARQMath-1 and ARQMath-2.")
    parser.add_argument("--preprocessing", default="with_latex", help="Index to search. Defaults to with_latex.")
    parser.add_argument("--k", type=int, default=10, help="nDCG cutoff. Defaults to 10.")
    args = parser.parse_args()

    from src.retrieval.bm25 import retrieve

    topics = {}
    for year in YEARS:
        topics.update(load(
            CLEANED_DIR / f"topics_{year}.jsonl",
            TOPIC_TRANSLATIONS.get(args.preprocessing) if args.preprocessing in TOPIC_TRANSLATIONS else None,
            with_latex=args.preprocessing == "with_latex",
        ))
    topics = list(topics.items())

    judgments = {}
    for year in YEARS:
        for line in (DATA_DIR / "qrels" / f"task1_{year}.tsv").read_text(encoding="utf-8").splitlines():
            qid, _, doc, grade = line.split("\t")
            judgments.setdefault(qid, {})[doc] = int(grade)
    topics = [(qid, text) for qid, text in topics if qid in judgments]
    if not topics:
        raise SystemExit("No ARQMath-1 or ARQMath-2 topics with judgments.")

    metric = f"ndcg@{args.k}"
    qrels = Qrels(judgments)
    runs = []
    for k1 in K1:
        for b in B:
            results = {}
            for qid, text in topics:
                hits = retrieve(text, args.preprocessing, args.k, k1, b)
                results[qid] = {str(doc): float(score) for doc, score in hits.items()}
            runs.append(Run(results, name=f"k1={k1} b={b}"))

    print(compare(qrels=qrels, runs=runs, metrics=[metric], make_comparable=True))
    best = max(runs, key=lambda run: evaluate(qrels, run, metric))
    print(f"best {best.name} {metric}={best.mean_scores[metric]:.4f}")


if __name__ == "__main__":
    main()
