"""Compare all running compatibility consumers against one local API window."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpp", type=Path)
    parser.add_argument("--php", default="php")
    args = parser.parse_args()
    root = os.environ.get("CAREOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")
    for decision, expected in (("--allow", "escalate"), ("--decline", "abstain")):
        review = json.loads(subprocess.check_output(
            [args.php, "integrations/php/review.php", "SYN-D-099", decision], timeout=12,
        ))
        assert review["result"]["status"] == expected, review
        assert bool(review["lookup"]) == (decision == "--allow")

    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(f"{root}/runs", params={"limit": 10})
        response.raise_for_status()
        runs = response.json()
        expected = {"runs": len(runs), "outcomes": dict(Counter(row["status"] for row in runs))}
        assert expected["runs"] >= 2
        actual = json.loads(subprocess.check_output([args.cpp.resolve()], timeout=10))
        assert actual == expected, ("C++", actual, expected)
        for name, port in (("Flask", 8002), ("Django", 8003), ("Laravel", 8004)):
            endpoint = f"http://127.0.0.1:{port}/summary"
            response = client.get(endpoint)
            response.raise_for_status()
            assert response.json() == expected, (name, response.json(), expected)
            assert client.post(endpoint).status_code == 405, name
    print("PHP review decisions and matching C++, Flask, Django and Laravel summaries passed")


if __name__ == "__main__":
    main()
