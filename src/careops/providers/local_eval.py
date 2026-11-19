"""Compare the two loopback-only policy answer configurations on development cases."""

import argparse
import json
import platform
import re
from pathlib import Path
from statistics import median
from time import perf_counter

import httpx

from careops.answer import GENERATION_SPEC, PROMPT_VERSION, load_answerer
from careops.answer_eval import load_development_cases
from careops.corpus import DEV_CASES, SOURCES, file_digest
from careops.embedding import load_model_spec


def evaluate(engine, cases):
    answerer = load_answerer(engine=engine)
    if engine == "ollama":
        url = answerer.generator.url.removesuffix("/api/chat")
        name = answerer.generator.model
        expected_name = name if ":" in name else name + ":latest"
        with httpx.Client(timeout=20, trust_env=False, base_url=url) as client:
            version = client.get("/api/version").raise_for_status().json()["version"]
            tags = client.get("/api/tags").raise_for_status().json()["models"]
        installed = next(row for row in tags if row["name"] == expected_name)
        model = {"runtime_version": version, "name": expected_name, "digest": installed["digest"],
                 "format": installed["details"]["format"],
                 "quantization": installed["details"]["quantization_level"]}
    else:
        from importlib.metadata import version
        model = {"runtime_version": version("transformers")}

    answerer.answer(cases[0]["question"])
    details, excerpts = [], {}
    for case in cases:
        start = perf_counter()
        reply = answerer.answer(case["question"])
        elapsed = (perf_counter() - start) * 1000
        expected = {(item["source_id"], item["pdf_page"])
                    for item in case["expected"].get("evidence", [])}
        citations = []
        for item in reply["citations"]:
            page = int(re.match(r"PDF page (\d+),", item["location"])[1])
            citations.append({"source_id": item["source_id"], "page": page,
                              "chunk_id": item["chunk_id"]})
        if len(citations) != int(reply["status"] == "answer"):
            raise RuntimeError("Policy answer status and citation count disagree")
        if citations:
            excerpts[case["case_id"]] = reply["answer"]
        details.append({
            "case_id": case["case_id"], "kind": case["kind"], "status": reply["status"],
            "reason": reply["reason"], "citations": citations,
            "reference_page_match": any((item["source_id"], item["page"]) in expected
                                        for item in citations),
            "latency_ms": round(elapsed, 2),
        })
    answerable = [row for row in details if row["kind"] == "policy"]
    negatives = [row for row in details if row["kind"] != "policy"]
    return {
        **model,
        "answerable": len(answerable),
        "answerable_answered": sum(row["status"] == "answer" for row in answerable),
        "answers_on_reference_page": sum(row["reference_page_match"] for row in answerable),
        "negative": len(negatives),
        "negative_abstained": sum(row["status"] == "abstain" for row in negatives),
        "p50_latency_ms": round(median(row["latency_ms"] for row in details), 2),
        "cases": details,
    }, excerpts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excerpts", type=Path, help="Optional local file for reviewing returned text")
    args = parser.parse_args()

    model = load_model_spec(GENERATION_SPEC)
    results, excerpts = {}, {}
    cases = load_development_cases()
    for engine in ("transformers", "ollama"):
        results[engine], excerpts[engine] = evaluate(engine, cases)
    report = {
        "split": "public development",
        "fixtures_sha256": file_digest(DEV_CASES), "sources_sha256": file_digest(SOURCES),
        "platform": platform.platform(), "python": platform.python_version(),
        "prompt_version": PROMPT_VERSION, "upstream_model": model["repository"],
        "revision": model["revision"], "weights_sha256": model["files"]["model.safetensors"]["sha256"],
        "engines": results,
    }
    if args.excerpts:
        args.excerpts.write_text(json.dumps(excerpts, ensure_ascii=False, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
