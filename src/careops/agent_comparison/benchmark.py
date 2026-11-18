"""Run one isolated framework against the fixed synthetic review comparison."""

import argparse
import asyncio
import json
import platform
import resource
from dataclasses import asdict
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

from jsonschema import Draft202012Validator

from careops.agent_comparison.task import CASES, CLAIM, POLICY, FixtureTools, load_fixture
from careops.corpus import DEV_CLAIMS, SOURCES, digest, file_digest

FRAMEWORKS = {
    "langgraph": "langgraph",
    "semantic_kernel": "semantic-kernel",
    "adk": "google-adk",
    "autogen": "autogen-core",
}
APPROVED_REPEATS = 7


async def evaluate(name):
    run = import_module(f"careops.agent_comparison.{name}").run
    fixture = load_fixture()
    claims, passage = fixture
    schema = json.loads(Path("schemas/agent_result.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    async def run_case(case):
        tools = FixtureTools(fixture, case)
        output, native = await run(case, tools)
        result = output["result"]
        validator.validate(result)
        calls = list(case.calls)
        expected_claim = claims.get(case.claim_id) if case.approved else None
        expected_lookup = expected_claim.model_dump() if expected_claim is not None else None
        expected_refs = [passage.chunk_id] if case.status == "review" else []
        expected_attempts = [(call, "timeout" if call == case.fail_tool and
                              calls[:idx + 1].count(call) <= case.failures
                              else "ok") for idx, call in enumerate(calls)]
        assert result["status"] == case.status and result["reason"] == case.reason, (name, case.name)
        assert result["tool_calls"] == calls and tools.events == expected_attempts, (name, case.name)
        assert output["lookup"] == expected_lookup, (name, case.name)
        assert result["citations"] == expected_refs, (name, case.name)
        if expected_refs:
            assert result["summary"].endswith(passage.text), (name, case.name)
        assert [event for event in native if event in (CLAIM, POLICY)] == calls, (name, case.name)
        return {
            "case": case.name, "status": result["status"], "reason": result["reason"],
            "citation_ids": result["citations"], "tool_outcomes": tools.events,
            "native_events": native, "result_schema_valid": True,
        }

    cases = [await run_case(case) for case in CASES]
    approved = CASES[0]
    timings = []
    for _ in range(APPROVED_REPEATS):
        start = perf_counter()
        await run_case(approved)
        timings.append((perf_counter() - start) * 1000)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "framework": name,
        "version": version(FRAMEWORKS[name]),
        "environment": {"python": platform.python_version(), "os": platform.platform(),
                        "pydantic": version("pydantic"), "pypdf": version("pypdf")},
        "inputs": {"source_manifest_sha256": file_digest(SOURCES),
                   "claims_sha256": file_digest(DEV_CLAIMS),
                   "case_definitions_sha256": digest(json.dumps(
                       [asdict(case) for case in CASES], sort_keys=True
                   ).encode())},
        "metrics": {
            "correct_cases": len(cases),
            "native_tool_trace_matches": len(cases),
            "schema_valid_cases": len(cases),
            "warm_approved_runs": len(timings),
            "warm_approved_p50_ms": round(median(timings), 3),
            "warm_approved_max_ms": round(max(timings), 3),
            "peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=FRAMEWORKS)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(args.framework)), indent=2))


if __name__ == "__main__":
    main()
