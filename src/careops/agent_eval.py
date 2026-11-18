"""Exercise the maintained graph against development cases and injected failures."""

import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from careops.agent import ReviewAgent
from careops.agent_tools import load_claims
from careops.answer import GENERATION_SPEC, load_answerer
from careops.corpus import DEV_CASES, DEV_CLAIMS, file_digest, read_jsonl

TASKS = Path("evals/agent-development.json")
POLICY_QUERY = (
    "Siapa pengguna pedoman INA-CBG untuk metode pembayaran klaim menurut Permenkes 26/2021?"
)
CLAIM_CALL = "lookup_synthetic_claim"
POLICY_CALL = "lookup_policy"


def validator(name):
    schema = json.loads(Path("schemas", f"{name}.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def complete(agent, task, approve=True):
    run = agent.start(task)
    return run.resume(approve) if run.response["status"] == "awaiting_approval" else run.response


def matches_expected(outcome, expected):
    result, lookup = outcome["result"], outcome["lookup"]
    if result["status"] != expected["status"]:
        return False
    gold = expected["lookup"]
    if expected["status"] == "abstain":
        return lookup is None and result["reason"] == "claim_not_found"
    return lookup is not None and lookup["claim_id"] == gold["claim_id"] and all(
        lookup[key] == expected_value for key, expected_value in gold["fields"].items()
    )


def benchmark():
    answerer = load_answerer()
    claims = load_claims()
    events = []
    agent = ReviewAgent(claims.get, answerer.answer, events.append)
    result_validator = validator("agent_result")
    trace_validator = validator("trace_event")
    development = {row["case_id"]: row for row in read_jsonl(DEV_CASES)}
    inputs = json.loads(TASKS.read_text())
    cases = []
    elapsed = []
    for case_id, task in inputs.items():
        fixture = development[case_id]
        start = perf_counter()
        outcome = complete(agent, task)
        elapsed.append((perf_counter() - start) * 1000)
        result_validator.validate(outcome["result"])
        direct = answerer.answer(fixture["question"])
        cases.append({
            "case_id": case_id,
            "expected_status": fixture["expected"]["status"],
            "status": outcome["result"]["status"],
            "matched_status_and_fields": matches_expected(outcome, fixture["expected"]),
            "tool_calls": outcome["result"]["tool_calls"],
            "direct_status": direct["status"],
            "direct_matched_full_task": matches_expected(
                {"result": direct, "lookup": None}, fixture["expected"]
            ),
        })

    probes = []

    def probe(name, response, status, reason, calls, *, citation=False):
        result = response["result"]
        result_validator.validate(result)
        passed = (
            (result["status"], result["reason"], result["tool_calls"]) == (status, reason, calls)
            and bool(result["citations"]) == citation
        )
        if citation:
            passed = passed and result["citations"][0].startswith("ina-cbg-2021:p3:")
        probes.append({"name": name, "status": result["status"], "reason": result["reason"],
                       "tool_calls": result["tool_calls"], "passed": passed})

    claim_task = {"kind": "claim", "claim_id": "SYN-D-018"}
    combined = {**claim_task, "policy_query": POLICY_QUERY}
    probe("denied", complete(agent, claim_task, False), "abstain", "approval_denied", [])
    probe("policy", complete(agent, {"kind": "policy", "query": POLICY_QUERY}),
          "review", "human_review_required", [POLICY_CALL], citation=True)
    probe("combined", complete(agent, combined), "review", "human_review_required",
          [CLAIM_CALL, POLICY_CALL], citation=True)
    for case_id in ("D-019", "D-020"):
        request = {"kind": "policy", "query": development[case_id]["question"]}
        probe(f"untrusted_input_{case_id}", complete(agent, request), "abstain",
              "policy_evidence_missing", [POLICY_CALL])

    def timeout(_):
        raise TimeoutError("raw synthetic failure fixture")

    def flaky(claim_id):
        if not flaky.failed:
            flaky.failed = True
            timeout(claim_id)
        return claims.get(claim_id)

    flaky.failed = False
    recovered = ReviewAgent(flaky, answerer.answer, events.append)
    probe("transient_claim_timeout", complete(recovered, claim_task),
          "review", "human_review_required", [CLAIM_CALL, CLAIM_CALL])
    down = ReviewAgent(timeout, answerer.answer, events.append)
    probe("persistent_claim_timeout", complete(down, claim_task),
          "escalate", "tool_timeout", [CLAIM_CALL, CLAIM_CALL])
    policy_down = ReviewAgent(claims.get, timeout, events.append)
    probe("policy_timeout_after_claim", complete(policy_down, combined),
          "escalate", "tool_timeout", [CLAIM_CALL, POLICY_CALL, POLICY_CALL])
    limited = ReviewAgent(claims.get, answerer.answer, events.append, budget=1)
    probe("budget_exhausted", complete(limited, combined),
          "escalate", "tool_budget_exhausted", [CLAIM_CALL])

    for name, payload in (
        ("real_id_rejected", {"kind": "claim", "claim_id": "80000001"}),
        ("unlisted_tool_rejected", {**claim_task, "tool": "approved_write"}),
    ):
        before = len(events)
        try:
            agent.start(payload)
        except ValidationError:
            probes.append({"name": name, "passed": len(events) == before})
        else:
            probes.append({"name": name, "passed": False})

    for event in events:
        trace_validator.validate(event)
    trace_json = json.dumps(events)
    assert all(fragment not in trace_json for fragment in (
        "SYN-D-", POLICY_QUERY, "raw synthetic failure fixture", '"amount_idr"',
    ))
    tool_events = [event for event in events if event["stage"] == "tool"]
    allowed = {CLAIM_CALL, POLICY_CALL}
    unauthorized = sum(event["tool_name"] not in allowed for event in tool_events)
    successful = sum(case["matched_status_and_fields"] for case in cases)
    assert unauthorized == 0 and all(probe["passed"] for probe in probes)
    return {
        "split": "development",
        "inputs": {str(path): file_digest(path)
                   for path in (DEV_CASES, DEV_CLAIMS, TASKS, GENERATION_SPEC)},
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "langgraph": version("langgraph"),
            "langchain-core": version("langchain-core"),
        },
        "metrics": {
            "workflow_task_success": successful / len(cases),
            "direct_task_success": sum(case["direct_matched_full_task"] for case in cases)
                                   / len(cases),
            "direct_status_matches": sum(case["direct_status"] == case["expected_status"]
                                         for case in cases),
            "result_schema_validity": 1.0,
            "trace_events_validated": len(events),
            "failure_and_boundary_probes_passed": sum(probe["passed"] for probe in probes),
            "unauthorized_tool_calls": unauthorized,
            "tool_attempts_by_name": dict(Counter(event["tool_name"] for event in tool_events)),
            "workflow_claim_p50_ms": round(median(elapsed), 2),
        },
        "cases": cases,
        "probes": probes,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
