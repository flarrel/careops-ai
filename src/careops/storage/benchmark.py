"""Run disposable storage checks and compare local vector search on development data."""

import argparse
import json
import math
import platform
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from careops.agent_tools import load_claims
from careops.corpus import DEV_CASES, DEV_CLAIMS, RAW, SOURCES, file_digest, load_sources, read_jsonl
from careops.embedding import E5Encoder, MODEL_DIRECTORY, load_model_spec
from careops.ingest import load_chunks
from careops.retrieval import DenseIndex, Hit, ranked
from careops.retrieval_eval import score_hits
from careops.storage import rows, vectors

K = 5
PASSES = 5


def latency(values):
    return {"requests": len(values), "p50_ms": round(median(values), 3),
            "p95_ms": round(sorted(values)[math.ceil(0.95 * len(values)) - 1], 3)}


def row_benchmark(directory, namespace):
    claims = [record.model_dump() for record in load_claims().values()]
    results = {}
    for engine in rows.ENGINES:
        result, timings = rows.exercise(engine, directory, namespace, claims)
        results[engine] = {**result, "range_query_latency": latency(timings)}
    return {"fixture_sha256": file_digest(DEV_CLAIMS), "engines": results}


def exact_ids(chunks, matrix, vector, source=None):
    scores = matrix @ vector
    selected = [(chunk, score) for chunk, score in zip(chunks, scores, strict=True)
                if source is None or chunk.source_id == source]
    return [hit.chunk.chunk_id for hit in ranked(
        [chunk for chunk, _ in selected], [score for _, score in selected], K,
    )]


def vector_queries(search, queries, cases, chunks, expected):
    results = []
    timings = []
    search(queries[0])
    for run in range(PASSES):
        for case, query, target in zip(cases, queries, expected, strict=True):
            start = perf_counter()
            ids = search(query)
            timings.append((perf_counter() - start) * 1000)
            if len(ids) != K or len(set(ids)) != K or any(key not in chunks for key in ids):
                raise RuntimeError("Vector search returned incomplete or unknown evidence IDs")
            if run == 0:
                metrics = score_hits([Hit(chunks[key], 0.0) for key in ids],
                                     case["expected"]["evidence"])
                results.append({"case_id": case["case_id"], **metrics,
                                "exact_overlap": len(set(ids) & set(target)) / K,
                                "ranked_ids": ids})
    return {
        "recall_at_5": round(mean(row["page_recall"] for row in results), 4),
        "mrr_at_5": round(mean(row["reciprocal_rank"] for row in results), 4),
        "source_hit_rate_at_5": round(mean(row["source_hit"] for row in results), 4),
        "mean_exact_overlap_at_5": round(mean(row["exact_overlap"] for row in results), 4),
        "query_latency": latency(timings),
        "cases": results,
    }


def verify_filter(search, query, source, expected, chunks):
    ids = search(query, source)
    if len(set(ids)) != K or any(chunks[key].source_id != source for key in ids):
        raise RuntimeError("Source filter leaked or missed eligible evidence")
    if set(ids) != set(expected):
        raise RuntimeError("Filtered vector results differ from exact search")
    if search(query, "missing-development-source"):
        raise RuntimeError("Missing metadata filter returned unrelated evidence")
    return {"source_id": source, "ranked_ids": ids, "missing_source_empty": True}


def vector_benchmark(directory, namespace, raw, model_directory):
    import numpy as np
    import torch

    sources = [row for row in load_sources() if row["case_split"] == "development"]
    chunks = load_chunks(sources, raw)
    cases = [case for case in read_jsonl(DEV_CASES) if case["kind"] == "policy"]
    encoder = E5Encoder(model_directory)
    dense = DenseIndex(chunks, encoder)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    queries = [encoder.query(case["question"]) for case in cases]
    source = cases[0]["expected"]["evidence"][0]["source_id"]
    expected = [exact_ids(chunks, dense.matrix, query) for query in queries]
    filtered = exact_ids(chunks, dense.matrix, queries[0], source)

    def exact(query, selected=None):
        return exact_ids(chunks, dense.matrix, query, selected)

    output = {
        "exact": {
            "server_version": np.__version__, "points": len(chunks), "index": "NumPy cosine",
            **vector_queries(exact, queries, cases, by_id, expected),
            "filter_acceptance": verify_filter(exact, queries[0], source, filtered, by_id),
        },
    }
    vector_queries_lists = [query.tolist() for query in queries]
    payloads = dense.matrix.tolist()
    for engine in vectors.ENGINES:
        with closing(vectors.connect(engine, directory, namespace)) as store:
            try:
                store.setup(dense.matrix.shape[1])
                start = perf_counter()
                for begin in range(0, len(chunks), vectors.BATCH):
                    end = begin + vectors.BATCH
                    store.add(chunks[begin:end], payloads[begin:end])
                store.add(chunks[:1], payloads[:1])
                index_ms = (perf_counter() - start) * 1000
                if store.count() != len(chunks):
                    raise RuntimeError(f"{engine}: indexed cardinality differs")

                def search(query, selected=None):
                    return store.search(query, K, source=selected)

                results = vector_queries(search, vector_queries_lists, cases, by_id, expected)
                if results["mean_exact_overlap_at_5"] < 0.95:
                    raise RuntimeError(f"{engine}: mean overlap with exact search is below 95%")
                output[engine] = {
                    "server_version": store.version(),
                    "points": len(chunks),
                    "index": {"chroma": "HNSW cosine", "qdrant": "exact cosine",
                              "weaviate": "flat cosine"}[engine],
                    "ingest_and_replay_ms": round(index_ms, 3),
                    **results,
                    "filter_acceptance": verify_filter(
                        search, vector_queries_lists[0], source, filtered, by_id,
                    ),
                }
            finally:
                store.cleanup()
    spec = load_model_spec()
    return {
        "source_manifest_sha256": file_digest(SOURCES),
        "development_cases_sha256": file_digest(DEV_CASES),
        "embedding_model": {key: spec[key] for key in ("repository", "revision")},
        "weights_sha256": spec["files"]["model.safetensors"]["sha256"],
        "device": "cpu", "torch_threads": torch.get_num_threads(),
        "dimensions": dense.matrix.shape[1], "numeric_payload_bytes": dense.payload_bytes,
        "policy_questions": len(cases), "top_k": K, "engines": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("rows", "vectors", "all"), default="all")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIRECTORY)
    args = parser.parse_args()
    token = uuid4().hex[:12]
    report = {
        "split": "public development",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "client_versions": {name: version(name) for name in (
                "chromadb", "httpx", "psycopg", "pymongo", "PyMySQL", "redis",
            )},
        },
    }
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        if args.track in ("rows", "all"):
            report["claims"] = row_benchmark(directory, f"claim_lab_{token}")
        if args.track in ("vectors", "all"):
            report["vector_search"] = vector_benchmark(
                directory, f"vector_lab_{token}", args.raw, args.model_dir,
            )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
