"""Evaluate frozen local methods against an explicitly supplied, checksum-verified holdout."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import resource
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from fastapi.testclient import TestClient
from pydantic import ValidationError

from careops.agent import ReviewAgent
from careops.agent_eval import complete, matches_expected, validator
from careops.agent_tools import load_claims
from careops.answer import (GENERATION_DIRECTORY, GENERATION_SPEC, PROMPT_VERSION, SELECT_PROMPT, VERIFY_PROMPT,
                            RETRIEVAL_K, LocalGenerator, PolicyAnswerer, load_answerer,
                            matches_scope, requested_scope)
from careops.answer_eval import abstained_baseline, prediction_digest, summarize
from careops.api import AnswerResponse, create_app
from careops.corpus import RAW, SEAL, SOURCES, file_digest, load_sources, read_jsonl, verify_seal
from careops.embedding import MODEL_DIRECTORY, load_model_spec
from careops.ingest import load_chunks
from careops.retrieval import retrieve
from careops.retrieval_eval import build_indexes, evaluate


def timing(values):
    ordered = sorted(values)
    return {"n": len(ordered), "p50_ms": round(median(ordered), 2),
            "p95_ms": round(ordered[math.ceil(.95 * len(ordered)) - 1], 2),
            "requests_per_second": round(len(ordered) * 1000 / sum(ordered), 2)}


def timed(fn, *args, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    return result, (perf_counter() - start) * 1000


def answer_rows(cases, answerer, maintained):
    rows = []
    for case in cases:
        query = case["question"]
        grounded, grounded_ms = timed(answerer.answer, query)
        baseline, baseline_ms = timed(answerer.generator.without_retrieval, query)
        unchanged, maintained_ms = timed(maintained.answer, query)
        for result in (grounded, unchanged):
            AnswerResponse.model_validate(result)
        scope = requested_scope(query)
        hits = retrieve(answerer.index, query, RETRIEVAL_K)["items"] if scope else []
        eligible = [hit for hit in hits if matches_scope(
            hit, answerer.sources[hit["source_id"]], scope
        )]
        expected = {(r["source_id"], r["pdf_page"]) for r in case["expected"].get("evidence", [])}

        def expected_page(item):
            return (item["source_id"], int(re.match(r"PDF page (\d+),", item["location"])[1])) in expected

        rows.append({
            "case_id": case["case_id"], "kind": case["kind"], "grounded": grounded,
            "baseline": {"status": "abstain" if abstained_baseline(baseline) else "answer",
                         "answer": baseline},
            "maintained": unchanged,
            "eligible_expected_page": any(expected_page(item) for item in eligible),
            "expected_page_in_shortlist": any(expected_page(item) for item in hits),
            "grounded_ms": grounded_ms, "baseline_ms": baseline_ms, "maintained_ms": maintained_ms,
        })
    return rows


def tool_eval(cases, directory, answerer):
    events = []
    claims = load_claims(directory / "claims.jsonl")
    agent = ReviewAgent(claims.get, answerer.answer, events.append)
    result_schema, event_schema = validator("agent_result"), validator("trace_event")
    statuses = Counter()
    checks = Counter()
    normal_timings = []
    first_lookup = cases[0]["expected"]["lookup"]["claim_id"]
    for case in cases:
        expected = case["expected"]
        claim_id = re.search(r"SYN-[A-Z]-[0-9]{3}", case["question"])[0]
        assert claim_id == expected["lookup"]["claim_id"]
        task = {"kind": "claim", "claim_id": claim_id}
        offset = len(events)
        run = agent.start(task)
        assert run.response["status"] == "awaiting_approval"
        checks["preapproval_tool_attempts"] += sum(e["stage"] == "tool" for e in events[offset:])
        outcome, duration = timed(run.resume, True)
        normal_timings.append(duration)
        result_schema.validate(outcome["result"])
        statuses[outcome["result"]["status"]] += 1
        checks["workflow_success"] += matches_expected(outcome, expected)
        checks["escalations_expected"] += expected["status"] == "escalate"
        checks["escalations_correct"] += (expected["status"] == outcome["result"]["status"]
                                          == "escalate")
        lookup = outcome["lookup"] or {}
        fields = expected["lookup"]["fields"]
        for key, value in fields.items():
            checks["fields_total"] += 1
            checks["fields_correct"] += lookup.get(key) == value
        baseline = answerer.answer(case["question"])
        checks["direct_success"] += matches_expected({"result": baseline, "lookup": None}, expected)

    probe = {"kind": "claim", "claim_id": first_lookup}
    denied = complete(agent, probe, False)
    result_schema.validate(denied["result"])

    def fail(_):
        raise TimeoutError("private-fault-marker")

    failed = complete(ReviewAgent(fail, answerer.answer, events.append), probe)
    result_schema.validate(failed["result"])
    malformed_rejected = 0
    for invalid in ({**probe, "tool": "unlisted"}, {**probe, "claim_id": "11111111"}):
        offset = len(events)
        try:
            agent.start(invalid)
        except ValidationError:
            malformed_rejected += len(events) == offset
    for event in events:
        event_schema.validate(event)
    trail = json.dumps(events)
    return {"cases": len(cases), "statuses": dict(statuses), "schema_valid_results": len(cases),
            "validated_trace_events": len(events), "attempts_by_tool": dict(Counter(
                e["tool_name"] for e in events if e["stage"] == "tool"
            )), **{key: checks[key] for key in (
                "workflow_success", "direct_success", "fields_correct", "fields_total",
                "escalations_correct", "escalations_expected", "preapproval_tool_attempts",
            )}, "denial_blocks_lookup": denied["result"]["reason"] == "approval_denied"
            and not denied["result"]["tool_calls"],
            "timeout_escalates_after_retry": failed["result"]["reason"] == "tool_timeout"
            and len(failed["result"]["tool_calls"]) == 2,
            "malformed_rejected": malformed_rejected, "unauthorized_tool_attempts": sum(
                e["stage"] == "tool" and e["tool_name"] != "lookup_synthetic_claim" for e in events
            ), "redaction_passed": not any(
                text in trail for text in ("SYN-H-", "private-fault-marker", '"amount_idr"')
            ), "approved_decision_timing": timing(normal_timings)}


def runtime_eval(raw, model):
    cases = [case for case in read_jsonl(Path("evals/development.jsonl")) if case["kind"] == "policy"]
    workload = [case["question"] for case in cases[:8]]
    marker = "private-malformed-marker"
    with TemporaryDirectory(prefix="careops-final-") as tmp:
        history = Path(tmp) / "runs.sqlite3"
        failed_canary = False

        def load():
            nonlocal failed_canary
            if not failed_canary:
                failed_canary = True
                raise OSError("private-fault-marker")
            return load_answerer(raw, model)

        with TestClient(create_app(load, history_path=history)) as client:
            fault = client.post("/answer", json={"query": workload[0]})
            ready, ready_ms = timed(client.get, "/ready")
            assert ready.status_code == 200
            measurements = {}
            outputs = {}
            for name in ("first", "repeat"):
                responses = [timed(client.post, "/answer", json={"query": item}) for item in workload]
                assert all(reply.status_code == 200 for reply, _ in responses)
                measurements[name] = timing([duration for _, duration in responses])
                outputs[name] = [reply.json() for reply, _ in responses]
            batch, batch_ms = timed(client.post, "/answers/batch", json={
                "queries": [workload[0], workload[1], workload[0]]
            })
            malformed = client.post("/answer", json={"query": [marker]})
            oversized = client.post("/answer", content="x" * 17000)
            assert batch.status_code == 200
            runs = client.get("/runs", params={"limit": 100}).json()
            metrics = client.get("/metrics").text
        traces = history.with_name("api-requests.jsonl").read_text()
        expected_rows = 1 + 2 * len(workload) + 2
        return {
            "transport": "in-process FastAPI TestClient", "fixture_split": "development",
            "recovery_after_injected_load_error": fault.status_code == 503 and ready.status_code == 200,
            "model_load_and_readiness_ms": round(ready_ms, 2), "timing": measurements,
            "batch_ms": round(batch_ms, 2),
            "batch_order_and_repeat_stable": batch.json() == [
                outputs["first"][0], outputs["first"][1], outputs["first"][0]
            ] and outputs["first"] == outputs["repeat"],
            "history_rows": len(runs), "expected_history_rows": expected_rows,
            "history_ids_unique": len({entry["run_id"] for entry in runs}) == len(runs),
            "malformed_status": malformed.status_code, "oversized_status": oversized.status_code,
            "redaction_passed": not any(term in traces + malformed.text + fault.text
                                       for term in [marker, "private-fault-marker", *workload]),
            "policy_counters": dict(re.findall(
                r'careops_policy_events_total\{event="([^"]+)"\} (\d+)', metrics
            )),
        }


def predict(sources, args, cases):
    import torch

    started = perf_counter()
    chunks = load_chunks(sources, args.raw)
    indexes, index_ms = timed(build_indexes, chunks, ("tfidf", "dense", "reranked"), args.embedding)
    only_policy = [case for case in cases if case["kind"] == "policy"]
    retrieval = {key: evaluate(index, only_policy) for key, index in indexes.items()}
    generator = LocalGenerator(args.model)
    answerer = PolicyAnswerer(indexes["tfidf"], generator, sources)
    development = [row for row in sources if row["case_split"] == "development"]
    maintained = PolicyAnswerer(build_indexes(load_chunks(development, args.raw),
                                             ("tfidf",), args.embedding)["tfidf"],
                               generator, development)
    questions = [case for case in cases if case["kind"] != "tool"]
    rows = answer_rows(questions, answerer, maintained)
    tools = tool_eval([case for case in cases if case["kind"] == "tool"], args.directory, answerer)
    runtime = runtime_eval(args.raw, args.model)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "split": "checksum-verified holdout",
        "seal_sha256": SEAL.read_text().strip(), "source_manifest_sha256": file_digest(SOURCES),
        "environment": {"os": platform.platform(), "machine": args.machine,
                        "python": platform.python_version(), "device": "cpu",
                        "torch_threads": torch.get_num_threads(),
                        "packages": {p: version(p) for p in ("torch", "transformers", "pypdf",
                                     "scikit-learn", "langgraph", "fastapi")}},
        "models": {key: {field: spec[field] for field in ("repository", "revision")} for key, spec in (
            ("generation", load_model_spec(GENERATION_SPEC)), ("embedding", load_model_spec())
        )},
        "configuration": {"indexed_documents": len(sources), "chunks": len(chunks),
                          "prompt": PROMPT_VERSION, "prompt_sha256": prediction_digest([SELECT_PROMPT, VERIFY_PROMPT]),
                          "answer_top_k": RETRIEVAL_K,
                          "retrieval_k": 5, "index_build_ms": round(index_ms, 2)},
        "retrieval": retrieval, "tools": tools, "runtime": runtime, "cases": rows,
        "process": {"peak_rss_bytes": usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
                    "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 2),
                    "wall_seconds": round(perf_counter() - started, 2)},
    }


def aggregate(predictions, judgments, cases):
    questions = [case for case in cases if case["kind"] != "tool"]
    report = {key: value for key, value in predictions.items() if key != "cases"}
    assert report["seal_sha256"] == SEAL.read_text().strip()
    reviewed = summarize(predictions, judgments, questions)
    report["generation"] = reviewed["metrics"]
    report["gates"] = {
        "dense_alone_improves_recall_or_mrr": any(
            report["retrieval"]["dense"][key] > report["retrieval"]["tfidf"][key]
            for key in ("recall_at_5", "mrr_at_5")
        ),
        "hybrid_improves_recall_or_mrr": any(
            report["retrieval"]["reranked"][key] > report["retrieval"]["tfidf"][key]
            for key in ("recall_at_5", "mrr_at_5")
        ),
        **reviewed["development_gates"],
        "synthetic_tool_task_success_at_least_0.80": (
            report["tools"]["workflow_success"] / report["tools"]["cases"] >= .80
        ),
        "all_tool_outputs_schema_valid": (
            report["tools"]["schema_valid_results"] == report["tools"]["cases"]
        ),
        "synthetic_structured_fields_accuracy_at_least_0.90": (
            report["tools"]["fields_correct"] / report["tools"]["fields_total"] >= .90
        ),
        "tool_and_trace_boundaries": (
            report["tools"]["unauthorized_tool_attempts"] == 0
            and report["tools"]["preapproval_tool_attempts"] == 0
            and report["tools"]["malformed_rejected"] == 2
            and report["tools"]["denial_blocks_lookup"]
            and report["tools"]["timeout_escalates_after_retry"]
            and report["tools"]["redaction_passed"]
        ),
        "runtime_history_and_controlled_errors": (
            report["runtime"]["history_ids_unique"] and report["runtime"]["redaction_passed"]
            and report["runtime"]["batch_order_and_repeat_stable"]
            and report["runtime"]["recovery_after_injected_load_error"]
            and report["runtime"]["history_rows"] == report["runtime"]["expected_history_rows"]
            and report["runtime"]["malformed_status"] == 422
            and report["runtime"]["oversized_status"] == 413
        ),
    }
    output = {row["case_id"]: row for row in predictions["cases"]}
    findings = Counter()
    for row in reviewed["cases"]:
        if row["kind"] != "policy":
            continue
        predicted = output[row["case_id"]]
        if row["grounded"] == "unsupported":
            findings["unsupported_returned"] += 1
        elif row["grounded"] == "supported":
            findings["supported"] += 1
        elif not predicted["expected_page_in_shortlist"]:
            findings["abstained_and_expected_page_missing_from_shortlist"] += 1
        elif not predicted["eligible_expected_page"]:
            findings["abstained_with_expected_page_excluded_by_scope"] += 1
        else:
            findings["abstained_despite_eligible_expected_page"] += 1
    report["generation"]["policy_failure_breakdown"] = dict(findings)
    report["generation"]["maintained_development_index"] = dict(Counter(
        ("abstained" if row["maintained"]["status"] == "abstain" else "returned_from_different_source_family")
        for row in output.values() if row["kind"] == "policy"
    ))
    report["generation"]["timing"] = {
        key: timing([row[f"{key}_ms"] for row in output.values()])
        for key in ("grounded", "baseline", "maintained")
    }
    report["case_counts"] = dict(Counter(case["kind"] for case in cases))
    report["retrieval"] = {
        name: {key: value for key, value in result.items() if key != "cases"}
        for name, result in report["retrieval"].items()
    }
    return report


def verify_replay(primary, other):
    def outputs(record):
        return [{key: value for key, value in row.items() if not key.endswith("_ms")}
                for row in record["cases"]]

    if primary["seal_sha256"] != other["seal_sha256"] or outputs(primary) != outputs(other):
        raise ValueError("The prediction replay does not match")
    methods = primary["retrieval"]
    if methods.keys() != other["retrieval"].keys() or any(
        records["cases"] != other["retrieval"][name]["cases"]
        for name, records in methods.items()
    ):
        raise ValueError("The retrieval replay does not match")
    return {"matching_answer_path_outcomes": 3 * len(primary["cases"]),
            "matching_ranked_queries": sum(len(m["cases"]) for m in methods.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("predict")
    benchmark.add_argument("--raw", type=Path, default=RAW)
    benchmark.add_argument("--model", type=Path, default=GENERATION_DIRECTORY)
    benchmark.add_argument("--embedding", type=Path, default=MODEL_DIRECTORY)
    benchmark.add_argument("--machine", default=platform.machine())
    score = commands.add_parser("summarize")
    score.add_argument("predictions", type=Path)
    score.add_argument("judgments", type=Path)
    score.add_argument("--replay", type=Path)
    args = parser.parse_args()
    sources = load_sources()
    verify_seal(sources, args.directory)
    cases = read_jsonl(args.directory / "cases.jsonl")
    if args.command == "predict":
        # The case-level file includes raw predictions and belongs beside the private holdout.
        if args.output.resolve().is_relative_to(Path.cwd().resolve()):
            parser.error("Write case-level predictions outside the working repository")
        report = predict(sources, args, cases)
    else:
        predictions = json.loads(args.predictions.read_text())
        report = aggregate(predictions, json.loads(args.judgments.read_text()), cases)
        if args.replay:
            report["reproduction"] = verify_replay(predictions, json.loads(args.replay.read_text()))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
