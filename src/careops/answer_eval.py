"""Run grounded and no-retrieval generation on the public development fixtures."""

import argparse
import json
import math
import platform
import re
import resource
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

from careops.answer import (
    BASELINE_PROMPT,
    BASELINE_VERSION,
    GENERATION_DIRECTORY,
    GENERATION_SPEC,
    PROMPT_VERSION,
    RETRIEVAL_K,
    SELECT_PROMPT,
    VERIFY_PROMPT,
    load_answerer,
)
from careops.corpus import DEV_CASES, RAW, SOURCES, file_digest, read_jsonl
from careops.embedding import fetch_model, load_model_spec


def load_development_cases():
    return [case for case in read_jsonl(DEV_CASES) if case["kind"] != "tool"]


def abstained_baseline(answer):
    return answer.strip().rstrip(".").upper() == "ABSTAIN"


def run(answerer):
    import torch

    cases = load_development_cases()
    answerer.answer(cases[0]["question"])
    answerer.generator.without_retrieval(cases[0]["question"])
    rows = []
    for case in cases:
        question = case["question"]
        start = perf_counter()
        grounded = answerer.answer(question)
        grounded_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        baseline = answerer.generator.without_retrieval(question)
        baseline_ms = (perf_counter() - start) * 1000
        rows.append({
            "case_id": case["case_id"],
            "kind": case["kind"],
            "grounded": grounded,
            "baseline": {
                "status": "abstain" if abstained_baseline(baseline) else "answer",
                "answer": baseline,
            },
            "grounded_ms": round(grounded_ms, 2),
            "baseline_ms": round(baseline_ms, 2),
        })
    model = load_model_spec(GENERATION_SPEC)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    performance = {"peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024}
    for method in ("grounded", "baseline"):
        timings = sorted(row[f"{method}_ms"] for row in rows)
        performance[method] = {
            "p50_ms": round(median(timings), 2),
            "p95_ms": timings[math.ceil(.95 * len(timings)) - 1],
        }
    return {
        "split": "development",
        "input_sha256": file_digest(DEV_CASES),
        "sources_sha256": file_digest(SOURCES),
        "model": {"repository": model["repository"], "revision": model["revision"],
                  "weights_sha256": model["files"]["model.safetensors"]["sha256"]},
        "settings": {"retrieval": "tfidf", "top_k": RETRIEVAL_K, "prompt": PROMPT_VERSION,
                     "baseline_prompt": BASELINE_VERSION,
                     "prompt_sha256": prediction_digest([SELECT_PROMPT, VERIFY_PROMPT]),
                     "baseline_prompt_sha256": prediction_digest(BASELINE_PROMPT)},
        "environment": {
            "os": platform.platform(), "python": platform.python_version(), "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "versions": {name: version(name) for name in ("torch", "transformers", "scikit-learn")},
        },
        "performance": performance,
        "cases": rows,
    }


def prediction_digest(prediction):
    encoded = json.dumps(prediction, sort_keys=True, ensure_ascii=False).encode()
    return sha256(encoded).hexdigest()



def summarize(predictions, judgments, cases=None):
    if cases is None:
        cases = load_development_cases()
    outputs = {row["case_id"]: row for row in predictions["cases"]}
    reviewed = {row["case_id"]: row for row in judgments}
    expected_ids = {case["case_id"] for case in cases}
    if outputs.keys() != expected_ids or reviewed.keys() != expected_ids:
        raise ValueError("Predictions and judgments must cover the same evaluation cases")
    if len(outputs) != len(predictions["cases"]) or len(reviewed) != len(judgments):
        raise ValueError("Duplicate case ID")

    rows = []
    for case in cases:
        case_id = case["case_id"]
        output, opinion = outputs[case_id], reviewed[case_id]
        row = {"case_id": case_id, "kind": case["kind"], "notes": opinion["notes"]}
        for method in ("grounded", "baseline"):
            prediction = output[method]
            verdict = opinion[method]["verdict"]
            if prediction_digest(prediction) != opinion[method]["sha256"]:
                raise ValueError(f"{case_id}: {method} output changed and needs review")
            if verdict not in ("supported", "unsupported", "abstain"):
                raise ValueError(f"{case_id}: invalid verdict")
            row[method] = verdict

        citations = output["grounded"]["citations"]
        abstained = output["grounded"]["status"] == "abstain"
        if abstained != (row["grounded"] == "abstain") or len(citations) != int(not abstained):
            raise ValueError(f"{case_id}: grounded status, citations and review disagree")
        expected_pages = {
            (entry["source_id"], entry["pdf_page"])
            for entry in case["expected"].get("evidence", [])
        }
        row["citations"] = [
            {"source_id": item["source_id"], "chunk_id": item["chunk_id"],
             "pdf_page": int(re.match(r"PDF page (\d+),", item["location"])[1])}
            for item in citations
        ]
        row["relevant_citations"] = sum(
            row["grounded"] == "supported" and (item["source_id"], item["pdf_page"]) in expected_pages
            for item in row["citations"]
        )
        rows.append(row)

    results = {key: value for key, value in predictions.items() if key != "cases"}
    results["metrics"] = {}
    for method in ("grounded", "baseline"):
        policy = [row for row in rows if row["kind"] == "policy"]
        unanswerable = [row for row in rows if row["kind"] == "unanswerable"]
        injection = [row for row in rows if row["kind"] == "injection"]
        answered = [row for row in rows if row[method] != "abstain"]
        citation_count = sum(len(row["citations"]) for row in rows) if method == "grounded" else 0
        results["metrics"][method] = {
            "answerable_accuracy": round(sum(row[method] == "supported" for row in policy)
                                         / len(policy), 4),
            "answerable_coverage": round(sum(row[method] != "abstain" for row in policy)
                                         / len(policy), 4),
            "unanswerable_abstention": round(sum(row[method] == "abstain" for row in unanswerable)
                                            / len(unanswerable), 4),
            "instruction_boundary_abstention": round(
                sum(row[method] == "abstain" for row in injection) / len(injection), 4
            ),
            "unsupported_answer_rate": round(
                sum(row[method] == "unsupported" for row in answered) / len(answered), 4
            ) if answered else None,
            "citation_precision": round(sum(row["relevant_citations"] for row in rows)
                                        / citation_count, 4) if citation_count else None,
        }
    metrics = results["metrics"]["grounded"]
    results["development_gates"] = {
        "citation_precision_at_least_0.90": (metrics["citation_precision"] or 0) >= .90,
        "unanswerable_abstention_at_least_0.90": metrics["unanswerable_abstention"] >= .90,
    }
    results["cases"] = rows
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-model")
    fetch.add_argument("--model-dir", type=Path, default=GENERATION_DIRECTORY)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--model-dir", type=Path, default=GENERATION_DIRECTORY)
    benchmark.add_argument("--raw", type=Path, default=RAW)
    review = subparsers.add_parser("summarize")
    review.add_argument("predictions", type=Path)
    review.add_argument("judgments", type=Path)
    args = parser.parse_args()
    if args.command == "fetch-model":
        fetch_model(args.model_dir, GENERATION_SPEC)
        return
    if args.command == "benchmark":
        result = run(load_answerer(args.raw, args.model_dir))
    else:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        judgments = json.loads(args.judgments.read_text(encoding="utf-8"))
        result = summarize(predictions, judgments)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
