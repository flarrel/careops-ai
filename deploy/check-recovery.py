"""Reproduce a missing-model fault and recover the same Docker API process."""

import argparse
import json
from pathlib import Path
from shutil import copyfile
from subprocess import check_output, run
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import httpx

from careops.answer import GENERATION_DIRECTORY
from careops.corpus import DEV_CASES, RAW, read_jsonl

WEIGHTS = "model.safetensors"


def docker(*args):
    return check_output(["docker", *args], text=True).strip()


def wait_until_live(client):
    deadline = monotonic() + 30
    while True:
        try:
            if client.get("/health", timeout=2).status_code == 200:
                return
        except httpx.RequestError:
            pass
        if monotonic() >= deadline:
            raise RuntimeError("Container did not become live")
        sleep(0.25)


def test(image, model, raw, temp_parent):
    query = read_jsonl(DEV_CASES)[0]["question"]
    with TemporaryDirectory(dir=temp_parent) as directory:
        model_dir = Path(directory) / "model"
        model_dir.mkdir()
        for item in model.iterdir():
            if item.is_file() and item.name != WEIGHTS:
                copyfile(item, model_dir / item.name)
        container = docker(
            "run", "--detach", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--user", "10001:10001",
            "--tmpfs", "/tmp:mode=1777", "--tmpfs", "/service/data/runtime:uid=10001,gid=10001,mode=0700",
            "--publish", "127.0.0.1::8000",
            "--mount", f"type=bind,src={raw},dst=/service/data/raw,readonly",
            "--mount", f"type=bind,src={model_dir},dst=/service/{GENERATION_DIRECTORY},readonly", image,
        )
        try:
            address = docker("port", container, "8000/tcp")
            with httpx.Client(base_url=f"http://{address}", timeout=60, trust_env=False) as client:
                wait_until_live(client)
                missing = client.get("/ready")
                failed = client.post("/answer", json={"query": query})
                if missing.status_code != 503 or failed.status_code != 503:
                    raise RuntimeError("Missing weights did not cause controlled unavailability")
                if failed.json() != {"detail": "Policy answer service unavailable"}:
                    raise RuntimeError("Unexpected detail in model failure response")
                staged = model_dir / (WEIGHTS + ".partial")
                copyfile(model / WEIGHTS, staged)
                staged.replace(model_dir / WEIGHTS)
                ready = client.get("/ready")
                answer = client.post("/answer", json={"query": query})
                ready.raise_for_status()
                answer.raise_for_status()
                if answer.json()["status"] != "answer" or not answer.json()["citations"]:
                    raise RuntimeError("The recovered model did not return a cited answer")
                statuses = [row["status"] for row in client.get("/runs").json()]
                if statuses != ["answer", "error"]:
                    raise RuntimeError("Recovery history does not match the requests")
                return {"missing_weights_ready_status": missing.status_code,
                        "missing_weights_answer_status": failed.status_code,
                        "restored_ready_status": ready.status_code,
                        "restored_answer_status": answer.status_code,
                        "cited_answer_after_restore": True, "same_container": True,
                        "history_newest_first": statuses}
        finally:
            run(["docker", "rm", "-f", container], check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="careops-ops13:local")
    parser.add_argument("--model", type=Path, default=GENERATION_DIRECTORY)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--temp-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(test(args.image, args.model.resolve(), args.raw.resolve(), args.temp_dir),
                     indent=2))


if __name__ == "__main__":
    main()
