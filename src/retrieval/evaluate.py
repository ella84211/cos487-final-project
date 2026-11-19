"""Evaluate TF-IDF, BM25, and the neural retrievers on the 2022 Task 1 topics.

TF-IDF and BM25 run on the first six ``INDEX_MAP`` entries. The neural retriever
runs on the last two. 2022 is the held-out test set.

Each run is scored twice. The plain metrics treat an unjudged hit as
non-relevant. The prime metrics (``ndcg'``, ``ndcg@10'``, and the rest) drop
hits that have no judgment for that topic, then score that condensed list.
Judged non-relevant documents stay. The cutoff is applied after condensing.

Runs are written as TSV files with columns ``qid``, ``docid``, ``rank``, and
``score``. After they are saved, a two-sided paired Student's t-test compares
every system on each metric. ``significance.txt`` is that table: a superscript
is a model the row scores higher than, with p < 0.05. ``significance.json``
stores each pair's scores, p-value, and whether that difference is significant.

    python -m src.retrieval.evaluate
"""

import json
import time

from tqdm import tqdm
from ranx import Qrels, Run, compare, evaluate

from src.utils import BM25_CONFIG, CLEANED_DIR, DATA_DIR, INDEX_MAP, MODEL_DIR, QRELS, TOPICS_MAP, load


YEAR = 2022
K = 100
METRICS = ["ndcg", "ndcg@5", "ndcg@10", "mrr", "precision@5", "precision@10"]
MAX_P = 0.05
RUNS_DIR = DATA_DIR / "runs"
METRICS_DIR = DATA_DIR / "metrics"


def _comparisons(report) -> list[dict]:
    """One record per unordered system pair and metric, with the t-test p-value."""
    records = []
    names = report.model_names
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for metric in report.metrics:
                stats = report.comparisons[left, right][metric]
                score_left = float(report.results[left][metric])
                score_right = float(report.results[right][metric])
                p_value = float(stats["p_value"])
                if score_left == score_right:
                    better = None
                else:
                    better = left if score_left > score_right else right
                records.append({
                    "metric": metric,
                    "systems": [left, right],
                    "scores": {left: score_left, right: score_right},
                    "p_value": None if p_value != p_value else p_value,
                    "significant": bool(stats["significant"]),
                    "better": better,
                })
    return records


def main():
    # retrievers
    from src.baseline.retrieval import retrieval as neural_retrieve
    from src.retrieval.bm25 import retrieve as bm25_retrieve
    from src.retrieval.tfidf import retrieve as tfidf_retrieve

    # systems to evaluate
    names = list(INDEX_MAP)
    systems = [(f"{kind}_{name}", name, kind) for name in names[:6] for kind in ("bm25", "tfidf")]
    systems += [(name, name, "neural") for name in names[6:]]
    missing = [name for name in dict.fromkeys(name for _, name, _ in systems) if not INDEX_MAP[name].is_file()]
    if missing:
        raise SystemExit(f"Missing indexes: {', '.join(missing)}. Build them before evaluating.")
    if any(name == "neural_ft" for _, name, _ in systems) and not (MODEL_DIR / "finetuned_model").exists():
        raise SystemExit("Missing fine-tuned model at models/finetuned_model.")

    # topics in the same representation as each index
    topic_cache = {}
    topics = {}
    for name in INDEX_MAP:
        style = "with_latex" if name in ("with_latex", "neural_pt", "neural_ft") else name
        if style not in topic_cache:
            loaded = {}
            loaded.update(load(
                CLEANED_DIR / f"topics_{YEAR}.jsonl",
                TOPICS_MAP.get(style),
                with_latex=style == "with_latex",
            ))
            topic_cache[style] = loaded
        topics[name] = topic_cache[style]

    # 2022 qrels
    qrels_dict = {}
    for line in QRELS[YEAR].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid, _, doc, grade = line.split("\t")
        qrels_dict.setdefault(qid, {})[doc] = int(grade)
    qrels = Qrels(qrels_dict)

    # tuned bm25 parameters, or the defaults
    bm25_params = {}
    if BM25_CONFIG.exists():
        name = None
        for line in BM25_CONFIG.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" "):
                name = line.split(":", 1)[0].strip()
                bm25_params[name] = {}
                continue
            key, value = line.split(":", 1)
            bm25_params[name][key.strip()] = float(value.strip())

    # output directories
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    runs = []
    prime_runs = []

    for system, name, kind in systems:
        # judged topics for this system
        queries = {qid: text for qid, text in topics[name].items() if qid in qrels_dict}
        if not queries:
            raise SystemExit(f"No judged topics for {system}.")
        params = bm25_params.get(name, {"k1": 1.2, "b": 0.75})
        results = {}
        latencies = {}
        # warm up the retriever before timing
        first = next(iter(queries.values()))
        if kind == "bm25":
            bm25_retrieve(first, name, K, params["k1"], params["b"])
        elif kind == "tfidf":
            tfidf_retrieve(first, name, K)
        else:
            neural_retrieve(first, "pretrained" if name == "neural_pt" else "finetuned", K)
        # retrieve and time each topic
        for qid, text in tqdm(queries.items(), desc=system):
            start = time.perf_counter()
            if kind == "bm25":
                hits = bm25_retrieve(text, name, K, params["k1"], params["b"])
            elif kind == "tfidf":
                hits = tfidf_retrieve(text, name, K)
            else:
                hits = neural_retrieve(text, "pretrained" if name == "neural_pt" else "finetuned", K)
            latencies[qid] = time.perf_counter() - start
            results[qid] = {str(doc): float(score) for doc, score in hits.items()}
        # keep every judged topic, even when nothing was retrieved
        for qid in qrels_dict:
            results.setdefault(qid, {})

        # save the run as tsv
        lines = ["qid\tdocid\trank\tscore"]
        for qid in sorted(results):
            ranked = sorted(results[qid].items(), key=lambda item: (-item[1], item[0]))
            for rank, (doc, score) in enumerate(ranked, start=1):
                lines.append(f"{qid}\t{doc}\t{rank}\t{score}")
        (RUNS_DIR / f"{system}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # plain metrics, unjudged hits count as non-relevant
        run = Run(results, name=system)
        runs.append(run)
        overall = evaluate(qrels, run, METRICS)
        # condense for prime metrics
        prime_run = Run(
            {qid: {doc: score for doc, score in hits.items() if doc in qrels_dict.get(qid, {})} for qid, hits in results.items()},
            name=system,
        )
        prime_runs.append(prime_run)
        prime_overall = evaluate(qrels, prime_run, METRICS)
        # per-query scores, latency, and index size
        scores = {}
        for metric in METRICS:
            scores[metric] = float(overall[metric])
            scores[f"{metric}'"] = float(prime_overall[metric])
        mean_latency = sum(latencies.values()) / len(latencies)
        index_path = INDEX_MAP[name]
        per_query = {}
        for qid, latency in latencies.items():
            per_query[qid] = {"latency_seconds": latency}
            for metric in METRICS:
                per_query[qid][metric] = float(run.scores[metric][qid])
                per_query[qid][f"{metric}'"] = float(prime_run.scores[metric][qid])
        payload = {
            "system": system,
            "index": str(index_path),
            "index_bytes": index_path.stat().st_size,
            "mean_latency_seconds": mean_latency,
            "metrics": scores,
            "queries": per_query,
        }
        (METRICS_DIR / f"{system}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary[system] = {
            "index_bytes": payload["index_bytes"],
            "mean_latency_seconds": mean_latency,
            **payload["metrics"],
        }
        print(f"{system} mean latency {mean_latency:.4f}s index {payload['index_bytes']} bytes")
        for metric, value in scores.items():
            print(f"  {metric} {value:.4f}")

    # summary of every system
    (METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # significance testing
    plain = compare(qrels=qrels, runs=runs, metrics=METRICS, stat_test="student", max_p=MAX_P)
    prime = compare(qrels=qrels, runs=prime_runs, metrics=METRICS, stat_test="student", max_p=MAX_P)
    report = (
        "Two-sided paired Student's t-test. A superscript is a model this row scores higher than, with p < 0.05.\n\n"
        + str(plain)
        + "\n\nPrime metrics, unjudged hits removed before scoring:\n\n"
        + str(prime)
        + "\n"
    )
    (METRICS_DIR / "significance.txt").write_text(report, encoding="utf-8")
    (METRICS_DIR / "significance.json").write_text(
        json.dumps(
            {
                "test": "two-sided paired Student's t-test",
                "max_p": MAX_P,
                "metrics": _comparisons(plain),
                "prime_metrics": _comparisons(prime),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"Wrote runs to {RUNS_DIR} and metrics to {METRICS_DIR}.")


if __name__ == "__main__":
    main()