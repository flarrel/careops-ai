"""Build local indexes, search with provenance, or evaluate public development questions."""

import argparse
import json
import math
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from careops.corpus import DEV_CASES, RAW, SOURCES, file_digest, load_sources, read_jsonl
from careops.embedding import E5Encoder, MODEL_DIRECTORY, fetch_model, load_model_spec
from careops.framework_search import LangChainIndex, LlamaIndex
from careops.ingest import load_chunks
from careops.retrieval import DenseIndex, LexicalIndex, RerankedIndex, retrieve

METHODS = ("tfidf", "dense", "reranked", "langchain", "llamaindex")
K = 5


def score_hits(hits, expected):
    pages = {(item["source_id"], item["pdf_page"]) for item in expected}
    expected_sources = {source for source, _ in pages}
    found = [(hit.chunk.source_id, hit.chunk.page) for hit in hits]
    first = next((rank for rank, page in enumerate(found, 1) if page in pages), None)
    return {
        "page_recall": len(pages.intersection(found)) / len(pages),
        "reciprocal_rank": 1 / first if first else 0.0,
        "source_hit": bool(expected_sources.intersection(source for source, _ in found)),
    }


def evaluate(index, cases):
    details = []
    timings = []
    index.search(cases[0]["question"], K)
    for case in cases:
        query = case["question"]
        start = perf_counter()
        hits = index.search(query, K)
        timings.append((perf_counter() - start) * 1000)
        metrics = score_hits(hits, case["expected"]["evidence"])
        details.append({
            "case_id": case["case_id"],
            **metrics,
            "hits": [
                {"source_id": hit.chunk.source_id, "pdf_page": hit.chunk.page,
                 "chunk_id": hit.chunk.chunk_id}
                for hit in hits
            ],
        })
    return {
        "recall_at_5": round(mean(row["page_recall"] for row in details), 4),
        "mrr_at_5": round(mean(row["reciprocal_rank"] for row in details), 4),
        "citation_source_hit_rate_at_5": round(mean(row["source_hit"] for row in details), 4),
        "p50_ms": round(median(timings), 2),
        "p95_ms": round(sorted(timings)[math.ceil(0.95 * len(timings)) - 1], 2),
        "numeric_payload_bytes": index.payload_bytes,
        "cases": details,
    }


def build_indexes(chunks, methods, model_directory):
    indexes = {}
    if "tfidf" in methods or "reranked" in methods:
        indexes["tfidf"] = LexicalIndex(chunks)
    if any(method != "tfidf" for method in methods):
        indexes["dense"] = DenseIndex(chunks, E5Encoder(model_directory))
    if "reranked" in methods:
        indexes["reranked"] = RerankedIndex(indexes["tfidf"], indexes["dense"])
    if "langchain" in methods:
        indexes["langchain"] = LangChainIndex(indexes["dense"])
    if "llamaindex" in methods:
        indexes["llamaindex"] = LlamaIndex(indexes["dense"])
    return {method: indexes[method] for method in methods}


def benchmark(chunks, indexes):
    cases = [case for case in read_jsonl(DEV_CASES) if case["kind"] == "policy"]
    env = {"python": platform.python_version(), "machine": platform.machine(),
           "os": platform.platform()}
    packages = ["pypdf", "scikit-learn", "numpy"]
    report = {
        "split": "development",
        "k": K,
        "questions": len(cases),
        "source_manifest_sha256": file_digest(SOURCES),
        "cases_sha256": file_digest(DEV_CASES),
        "chunks_by_source": dict(Counter(chunk.source_id for chunk in chunks)),
        "environment": env,
        "methods": {},
    }
    if set(indexes) != {"tfidf"}:
        spec = load_model_spec()
        report["embedding"] = {
            "model": spec["repository"], "revision": spec["revision"],
            "weights_sha256": spec["files"]["model.safetensors"]["sha256"],
        }
        import torch

        env.update(device="cpu", torch_threads=torch.get_num_threads())
        packages.extend(["torch", "transformers"])
    packages.extend(
        library for method, library in (("langchain", "langchain-core"),
                                        ("llamaindex", "llama-index-core"))
        if method in indexes
    )
    env["versions"] = {package: version(package) for package in packages}
    for name, index in indexes.items():
        report["methods"][name] = evaluate(index, cases)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-model")
    fetch.add_argument("--model-dir", type=Path, default=MODEL_DIRECTORY)
    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--method", choices=METHODS, default="tfidf")
    search.add_argument("--top-k", type=int, default=K)
    evaluation = subparsers.add_parser("benchmark")
    evaluation.add_argument("--methods", choices=METHODS, nargs="+", default=METHODS)
    for subparser in (search, evaluation):
        subparser.add_argument("--raw", type=Path, default=RAW)
        subparser.add_argument("--model-dir", type=Path, default=MODEL_DIRECTORY)
    args = parser.parse_args()
    if args.command == "fetch-model":
        fetch_model(args.model_dir)
        return

    sources = [row for row in load_sources() if row["case_split"] == "development"]
    chunks = load_chunks(sources, args.raw)
    if args.command == "search":
        indexes = build_indexes(chunks, [args.method], args.model_dir)
        result = retrieve(indexes[args.method], args.query, args.top_k)
    else:
        indexes = build_indexes(chunks, args.methods, args.model_dir)
        result = benchmark(chunks, indexes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
