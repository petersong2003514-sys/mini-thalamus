"""BEIR retrieval evaluation for the Mini Thalamus RAG stack.

The runner can load the standard BEIR zip layout directly:

    <dataset>/corpus.jsonl
    <dataset>/queries.jsonl
    <dataset>/qrels/<split>.tsv

It supports official downloads from the TU-Darmstadt BEIR mirror, an offline
``--smoke`` fixture, and both embedding and BM25 retrieval.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import zipfile


EVALS_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVALS_ROOT, "external", "beir_data")
REPORT_DIR = os.path.join(EVALS_ROOT, "results")
DEFAULT_DATASETS = ["nfcorpus", "scifact"]
BEIR_MIRROR = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_qrels(path):
    qrels = {}
    if not os.path.exists(path):
        return qrels
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("query-id"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0], parts[1], parts[2]
            if int(score) <= 0:
                continue
            qrels.setdefault(qid, set()).add(doc_id)
    return qrels


def load_dataset_dir(dataset, split="test", data_dir=DATA_DIR):
    root = os.path.join(data_dir, dataset)
    corpus = {
        row["_id"]: f"{row.get('title', '')}\n{row.get('text', '')}".strip()
        for row in read_jsonl(os.path.join(root, "corpus.jsonl"))
    }
    qrels = read_qrels(os.path.join(root, "qrels", f"{split}.tsv"))
    queries = {
        row["_id"]: row.get("text", row.get("title", "")).strip()
        for row in read_jsonl(os.path.join(root, "queries.jsonl"))
        if row["_id"] in qrels
    }
    return corpus, queries, qrels


def download_dataset(dataset, data_dir=DATA_DIR):
    url = f"{BEIR_MIRROR}/{dataset}.zip"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{dataset}.zip")
    print(json.dumps({"downloading": url}, ensure_ascii=False))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mini-thalamus-eval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
    with open(zip_path, "wb") as handle:
        handle.write(payload)
    target = os.path.abspath(data_dir)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            resolved = os.path.abspath(
                os.path.join(data_dir, member.filename)
            )
            if not resolved.startswith(target + os.sep):
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        archive.extractall(data_dir)
    return zip_path


def _embed_texts(texts, model_name):
    try:
        import pyarrow  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            f"embedding unavailable ({exc}); use --bm25 or install "
            "sentence-transformers"
        )
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def _bm25_retrieve(corpus, queries, top_k):
    from rank_bm25 import BM25Okapi

    corpus_ids = list(corpus)
    tokenized = [corpus[doc_id].lower().split() for doc_id in corpus_ids]
    bm25 = BM25Okapi(tokenized)
    ranked = {}
    for qid, query in queries.items():
        scores = bm25.get_scores(query.lower().split())
        order = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )
        ranked[qid] = [corpus_ids[index] for index in order[:top_k]]
    return ranked


def _embed_retrieve(corpus, queries, top_k, model_name):
    import numpy as np

    corpus_ids = list(corpus)
    corpus_vectors = _embed_texts(
        [corpus[doc_id] for doc_id in corpus_ids],
        model_name,
    )
    ranked = {}
    for qid, query in queries.items():
        query_vector = _embed_texts([query], model_name)[0]
        scores = np.dot(corpus_vectors, query_vector)
        order = scores.argsort()[::-1][:top_k]
        ranked[qid] = [corpus_ids[index] for index in order]
    return ranked


def _dcg(rel, k):
    return sum(
        rel[index] / math.log2(index + 2)
        for index in range(min(k, len(rel)))
    )


def _ndcg(predicted, relevant, k):
    rel = [1 if doc_id in relevant else 0 for doc_id in predicted[:k]]
    ideal = sorted(rel, reverse=True)
    denominator = _dcg(ideal, k)
    if denominator == 0:
        return 0.0
    return _dcg(rel, k) / denominator


def _mrr(predicted, relevant):
    for index, doc_id in enumerate(predicted[:10], start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def evaluate(corpus, queries, qrels, top_k=10, bm25=False, encoder=None):
    if bm25:
        ranked = _bm25_retrieve(corpus, queries, top_k)
    else:
        ranked = _embed_retrieve(corpus, queries, top_k, encoder)

    records = []
    for qid, query in queries.items():
        relevant = qrels.get(qid, set())
        predicted = ranked.get(qid, [])
        records.append(
            {
                "qid": qid,
                "query": query[:200],
                "predicted": predicted,
                "relevant": sorted(relevant),
                "ndcg_at_10": round(_ndcg(predicted, relevant, 10), 4),
                "mrr_at_10": round(_mrr(predicted, relevant), 4),
                "recall_at_10": round(
                    len(set(predicted[:10]) & relevant) / len(relevant)
                    if relevant
                    else 0.0,
                    4,
                ),
            }
        )
    total = len(records)
    metrics = {
        "case_count": total,
        "ndcg_at_10": round(
            sum(r["ndcg_at_10"] for r in records) / total, 4
        ) if total else 0,
        "mrr_at_10": round(
            sum(r["mrr_at_10"] for r in records) / total, 4
        ) if total else 0,
        "recall_at_10": round(
            sum(r["recall_at_10"] for r in records) / total, 4
        ) if total else 0,
    }
    return metrics, records


def write_report(dataset, metrics, records, output_dir=REPORT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = os.path.join(output_dir, f"beir-{dataset}-{stamp}")
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": "BEIR",
                "dataset": dataset,
                "metrics": metrics,
                "records": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    lines = [
        "# BEIR Retrieval Evaluation",
        "",
        f"- Dataset: {dataset}",
        f"- Cases: {metrics['case_count']}",
        f"- nDCG@10: {metrics['ndcg_at_10']}",
        f"- MRR@10: {metrics['mrr_at_10']}",
        f"- Recall@10: {metrics['recall_at_10']}",
        "",
        "| qid | nDCG@10 | MRR@10 | recall@10 |",
        "|---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['qid']} | {record['ndcg_at_10']} | "
            f"{record['mrr_at_10']} | {record['recall_at_10']} |"
        )
    with open(stem + ".md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return stem


def write_smoke_fixture(data_dir):
    root = os.path.join(data_dir, "_smoke")
    os.makedirs(os.path.join(root, "qrels"), exist_ok=True)
    corpus = [
        {"_id": "d1", "title": "Paris", "text": "Paris is the capital of France."},
        {"_id": "d2", "title": "Berlin", "text": "Berlin is the capital of Germany."},
        {"_id": "d3", "title": "Rome", "text": "Rome is the capital of Italy."},
    ]
    queries = [
        {"_id": "q1", "text": "capital of France"},
        {"_id": "q2", "text": "capital of Germany"},
    ]
    qrels = [
        ("q1", "d1", "1"),
        ("q2", "d2", "1"),
    ]
    with open(os.path.join(root, "corpus.jsonl"), "w", encoding="utf-8") as handle:
        for row in corpus:
            handle.write(json.dumps(row) + "\n")
    with open(os.path.join(root, "queries.jsonl"), "w", encoding="utf-8") as handle:
        for row in queries:
            handle.write(json.dumps(row) + "\n")
    with open(os.path.join(root, "qrels", "test.tsv"), "w", encoding="utf-8") as handle:
        handle.write("query-id corpus-id score\n")
        for row in qrels:
            handle.write("\t".join(row) + "\n")
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=",".join(DEFAULT_DATASETS),
        help="comma-separated BEIR datasets",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=REPORT_DIR)
    parser.add_argument(
        "--encoder",
        default=os.environ.get(
            "BEIR_ENCODER",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )
    parser.add_argument("--bm25", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        root = write_smoke_fixture(args.data_dir)
        corpus, queries, qrels = load_dataset_dir(
            "_smoke",
            split=args.split,
            data_dir=args.data_dir,
        )
    else:
        datasets = [
            item.strip() for item in args.dataset.split(",") if item.strip()
        ]
        if args.download:
            for dataset in datasets:
                download_dataset(dataset, data_dir=args.data_dir)
        if args.check_data:
            summary = {}
            for dataset in datasets:
                corpus, queries, qrels = load_dataset_dir(
                    dataset,
                    split=args.split,
                    data_dir=args.data_dir,
                )
                summary[dataset] = {
                    "corpus": len(corpus),
                    "queries": len(queries),
                    "qrels": len(qrels),
                }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        corpus, queries, qrels = load_dataset_dir(
            datasets[0],
            split=args.split,
            data_dir=args.data_dir,
        )

    if args.limit:
        queries = dict(list(queries.items())[: args.limit])
    metrics, records = evaluate(
        corpus,
        queries,
        qrels,
        top_k=args.top_k,
        bm25=args.bm25,
        encoder=args.encoder,
    )
    dataset_label = "_smoke" if args.smoke else datasets[0]
    stem = write_report(
        dataset_label,
        metrics,
        records,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {"report": stem + ".json", "dataset": dataset_label, "metrics": metrics},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
