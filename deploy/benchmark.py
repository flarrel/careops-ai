"""Measure the real local CareOps API with fixed public development requests."""

import argparse
import json
import math
import platform
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from time import perf_counter

import httpx

from careops.corpus import DEV_CASES, file_digest, read_jsonl

CASE_IDS = ("D-001", "D-002", "D-006", "D-012")
WORKERS = 4


def metrics(client):
    response = client.get("/metrics")
    response.raise_for_status()
    return {line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1])
            for line in response.text.splitlines() if line and not line.startswith("#")}


def event(values, label):
    return int(values.get(f'careops_policy_events_total{{event="{label}"}}', 0))


def phase(client, requests, *, workers):
    def call(item):
        path, body = item
        start = perf_counter()
        reply = client.post(path, json=body)
        elapsed = (perf_counter() - start) * 1000
        reply.raise_for_status()
        return elapsed, reply.json()

    before = metrics(client)
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(call, requests))
    wall = perf_counter() - start
    after = metrics(client)
    times = sorted(duration for duration, _ in results)
    output = [reply for _, reply in results]
    answers = [answer for reply in output for answer in (reply if isinstance(reply, list) else [reply])]
    return {
        "requests": len(requests), "questions": len(answers),
        "outcomes": dict(Counter(answer["status"] for answer in answers)),
        "p50_ms": round(median(times), 2), "p95_ms": round(times[math.ceil(.95 * len(times)) - 1], 2),
        "requests_per_second": round(len(requests) / wall, 2),
        "input_bytes": sum(len(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode())
                           for _, body in requests),
        "process_cpu_ms": round(1000 * (after["careops_process_cpu_seconds_total"] -
                                         before["careops_process_cpu_seconds_total"]), 2),
        "peak_process_rss_bytes": int(after["careops_process_peak_rss_bytes"]),
        "policy_events": {key: event(after, key) - event(before, key) for key in (
            "cache_hit", "cache_miss", "scope_abstain", "model_answer",
        )},
    }, output


def validate_limits(client):
    header = {"X-Request-ID": "fixture-should-not-be-logged"}
    oversize = client.post("/answer", content=b"fixture-should-not-be-logged" * 900, headers=header)
    invalid = client.post("/answer", json={"query": "fixture-should-not-be-logged", "extra": True})
    if (oversize.status_code, invalid.status_code) != (413, 422):
        raise RuntimeError("Request size and shape limits failed")
    if any("fixture-should-not-be-logged" in response.text or
           "fixture-should-not-be-logged" in response.headers["x-request-id"]
           for response in (oversize, invalid)):
        raise RuntimeError("Request input reached response metadata")
    return {"over_limit_status": 413, "invalid_shape_status": 422,
            "caller_request_id_rejected": True}


def benchmark(port):
    cases = {case["case_id"]: case["question"] for case in read_jsonl(DEV_CASES)
             if case["case_id"] in CASE_IDS}
    questions = [cases[key] for key in CASE_IDS]
    requests = [("/answer", {"query": query}) for query in questions]
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60, trust_env=False) as client:
        client.get("/ready").raise_for_status()
        report = {}
        report["initial_queries"], original = phase(client, requests, workers=WORKERS)
        report["repeat_queries"], repeated = phase(client, requests * 3, workers=WORKERS)
        if repeated != original * 3:
            raise RuntimeError("Repeated answers differ from their first results")
        no_scope = "No policy edition was supplied."
        batch = questions + [questions[0], no_scope, no_scope, questions[1]]
        report["mixed_batch"], responses = phase(
            client, [("/answers/batch", {"queries": batch})], workers=1,
        )
        history = client.get("/runs?limit=100")
        history.raise_for_status()
        history = history.json()
        ids = {row["run_id"] for row in history}
        if len(ids) != len(history) or len(history) != len(requests) * 4 + len(set(batch)):
            raise RuntimeError("Unexpected number of stored run summaries")
        if responses[0][:4] != original or responses[0][5]["reason"] != "scope_required":
            raise RuntimeError("Batch order or scope routing changed a result")
        if report["initial_queries"]["policy_events"]["model_answer"] != 4:
            raise RuntimeError("First queries did not exercise four model calls")
        if report["repeat_queries"]["policy_events"]["cache_hit"] != 12:
            raise RuntimeError("Repeat queries did not exercise twelve cache hits")
        if report["mixed_batch"]["policy_events"]["scope_abstain"] != 1:
            raise RuntimeError("Repeated out-of-scope inputs did not deduplicate")
        report["stored_run_summaries"] = len(history)
        report["security"] = validate_limits(client)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--target", choices=("docker", "kubernetes"), required=True)
    args = parser.parse_args()
    result = {
        "target": args.target, "host": platform.platform(), "python": platform.python_version(),
        "fixture_sha256": file_digest(DEV_CASES), "case_ids": CASE_IDS, "concurrency": WORKERS,
        "results": benchmark(args.port),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
