# Synthetic claim review agent

The local LangGraph workflow provides two read-only tools. `lookup_synthetic_claim` reads one project-owned fixture. `lookup_policy` uses the local cited-answer service. Both tools validate typed arguments. The graph routes on the caller's explicit task and each tool's structured outcome. Questions and policy excerpts stay data throughout the workflow.

A claim task pauses before the lookup. The caller can approve that read or decline it. After approval, the workflow checks fixture attachments, service codes and any simulation limit supplied with the request. An optional policy question runs after the fixture checks pass. The result is a human-review checklist, an abstention, or an escalation. Approval opens the fixture lookup for that run. Review results carry no payment or clinical authority.

Each run allows at most three tool attempts. A transient I/O failure gets one retry within that budget. A persistent failure, missing policy evidence for a combined task, or failed checklist routes to the appropriate human follow-up. Typed input rejects real-looking identifiers and extra fields. An in-memory checkpoint holds the approval pause for the local process.

## Run locally

Run from the repository root with Python 3.12.

```sh
python3 -m pip install -e '.[agent]'
python3 -m careops.agent_cli claim SYN-D-018
python3 -m careops.agent_cli claim SYN-D-013 --max-amount-idr 1800000 --approve
```

The first command after installation prompts for approval. Use `--deny` to decline explicitly. The optional amount limit is a simulation input and has no policy meaning.

To look up policy evidence, install `.[agent,rag,dev]` and prepare the local corpus and Qwen cache using the [answer service guide](grounded-answers.md).

```sh
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python3 -m careops.agent_cli policy \
  "Siapa pengguna pedoman INA-CBG untuk metode pembayaran klaim menurut Permenkes 26/2021?"
```

Pass `--policy-query` to a claim task to include the optional policy lookup. Claim-only tasks can run with the fixtures and the agent dependencies. The policy tool loads the model on its first call.

The CLI writes redacted JSONL to `data/runtime/agent-traces.jsonl`. Events carry a run ID, time, stage, duration, outcome, and relevant tool names, attempt counts, fixed error codes or public citation IDs. The request fingerprint is keyed per process. Logs omit questions, claim identifiers, lookup values, excerpts, and raw exceptions. An approval event records the decision without recording caller identity. The local trace remains available for inspection after the in-memory run finishes.

## Development evaluation

```sh
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python3 -m careops.agent_eval
```

The [development result](../evals/results/agent-development.json) records matched statuses and fields for all five claim tasks. The direct policy-answer comparator abstained on all five. It matched the surface status for the missing-claim case and could not establish fixture absence without the lookup. Each claim task uses a typed operator input from the [development task map](../evals/agent-development.json). The benchmark uses the original questions for the direct comparator.

All 11 additional probes passed, including denied approval, real local policy lookup, a combined two-tool task, injected instructions, malformed inputs, retries, and budget exhaustion. The run validated 55 trace events and recorded zero unauthorized tool calls. Injected timeout probes establish handling of reported tool failures. Wall-clock cancellation for a blocking local model remains open. The benchmark covers development fixtures only. The existing policy evidence coverage limits the combined path.
