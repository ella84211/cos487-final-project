"""Evaluate TF-IDF, BM25, and the neural retrievers on the 2022 Task 1 topics.

TF-IDF and BM25 run on the first six ``INDEX_MAP`` entries. The neural retriever
runs on the last two. 2022 is the held-out test set.

    python -m src.retrieval.evaluate
"""

import json
import time

from tqdm import tqdm
from ranx import Qrels, Run, evaluate

from src.utils import BM25_CONFIG, CLEANED_DIR, DATA_DIR, INDEX_MAP, MODEL_DIR, QRELS, TOPICS_MAP, load


YEAR = 2022
K = 100
METRICS = ["ndcg", "ndcg@5", "ndcg@10", "mrr", "precision@5", "precision@10"]
RUNS_DIR = DATA_DIR / "runs"
METRICS_DIR = DATA_DIR / "metrics"


def main():
    from src.baseline.retrieval import retrieval as neural_retrieve
    from src.retrieval.bm25 import retrieve as bm25_retrieve
    from src.retrieval.tfidf import retrieve as tfidf_retrieve

    names = list(INDEX_MAP)
    systems = [(f"{kind}_{name}", name, kind) for name in names[:6] for kind in ("bm25", "tfidf")]
    systems += [(name, name, "neural") for name in names[6:]]
    missing = [name for name in dict.fromkeys(name for _, name, _ in systems) if not INDEX_MAP[name].is_file()]
    if missing:
        raise SystemExit(f"Missing indexes: {', '.join(missing)}. Build them before evaluating.")
    if any(name == "neural_ft" for _, name, _ in systems) and not (MODEL_DIR / "finetuned_model").exists():
        raise SystemExit("Missing fine-tuned model at models/finetuned_model.")

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

    qrels_dict = {}
    for line in QRELS[YEAR].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid, _, doc, grade = line.split("\t")
        qrels_dict.setdefault(qid, {})[doc] = int(grade)
    qrels = Qrels(qrels_dict)

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

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for system, name, kind in systems:
        queries = {qid: text for qid, text in topics[name].items() if qid in qrels_dict}
        if not queries:
            raise SystemExit(f"No judged topics for {system}.")
        params = bm25_params.get(name, {"k1": 1.2, "b": 0.75})
        results = {}
        latencies = {}
        first = next(iter(queries.values()))
        if kind == "bm25":
            bm25_retrieve(first, name, K, params["k1"], params["b"])
        elif kind == "tfidf":
            tfidf_retrieve(first, name, K)
        else:
            neural_retrieve(first, "pretrained" if name == "neural_pt" else "finetuned", K)
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
        for qid in qrels_dict:
            results.setdefault(qid, {})

        run = Run(results, name=system)
        run.save(str(RUNS_DIR / f"{system}.trec"))
        overall = evaluate(qrels, run, METRICS)
        mean_latency = sum(latencies.values()) / len(latencies)
        index_path = INDEX_MAP[name]
        per_query = {}
        for qid, latency in latencies.items():
            per_query[qid] = {"latency_seconds": latency}
            for metric in METRICS:
                per_query[qid][metric] = float(run.scores[metric][qid])
        payload = {
            "system": system,
            "index": str(index_path),
            "index_bytes": index_path.stat().st_size,
            "mean_latency_seconds": mean_latency,
            "metrics": {metric: float(overall[metric]) for metric in METRICS},
            "queries": per_query,
        }
        (METRICS_DIR / f"{system}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary[system] = {
            "index_bytes": payload["index_bytes"],
            "mean_latency_seconds": mean_latency,
            **payload["metrics"],
        }
        print(f"{system} mean latency {mean_latency:.4f}s index {payload['index_bytes']} bytes")
        for metric in METRICS:
            print(f"  {metric} {payload['metrics'][metric]:.4f}")

    (METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote runs to {RUNS_DIR} and metrics to {METRICS_DIR}.")


if __name__ == "__main__":
    main()
