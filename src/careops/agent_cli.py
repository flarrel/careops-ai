"""Run a local policy lookup or approve read access to one synthetic claim."""

import argparse
import json
from functools import cache
from pathlib import Path

from pydantic import ValidationError

from careops.agent import load_agent
from careops.agent_trace import TRACE_PATH, JsonlTrace
from careops.answer import load_answerer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=TRACE_PATH)
    commands = parser.add_subparsers(dest="kind", required=True)
    policy = commands.add_parser("policy")
    policy.add_argument("query")
    claim = commands.add_parser("claim")
    claim.add_argument("claim_id")
    claim.add_argument("--max-amount-idr", type=int)
    claim.add_argument("--policy-query")
    decision = claim.add_mutually_exclusive_group()
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--deny", action="store_true")
    options = vars(parser.parse_args())

    @cache
    def answerer():
        return load_answerer()

    trace = JsonlTrace(options.pop("trace"))
    approve = options.pop("approve", False)
    deny = options.pop("deny", False)
    agent = load_agent(lambda query: answerer().answer(query), trace)
    try:
        run = agent.start(options)
    except ValidationError:
        parser.error(
            "Invalid task. Use a synthetic claim ID or a policy question of 1–500 characters."
        )
    if run.response["status"] == "awaiting_approval":
        if not approve and not deny:
            try:
                reply = input("Read this synthetic claim for human review? [y/N] ")
                approve = reply.strip().casefold() in {"y", "yes"}
            except EOFError:
                approve = False
        run.resume(approve)
    print(json.dumps(run.response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
