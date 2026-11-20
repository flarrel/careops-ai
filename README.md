# CareOps AI

CareOps AI is a local-first demonstrator for answering healthcare policy and synthetic-claims questions with source citations and safe abstention.

The maintained path is intentionally small:

```text
source files + synthetic claims -> ingestion -> retrieval -> evidence packet
                                                        |              |
                                                 cited answer     read-only review tools
                                                        \              /
                                                        FastAPI
                                                          |
                                                      Streamlit
```

The project uses provenance-recorded public policy material and project-owned synthetic records. It is not medical advice, a clinical decision system, a live claims processor, or a substitute for professional review.

## Run the application

The [local application guide](docs/application.md) starts FastAPI, the Streamlit admin, and the Next.js client for policy and synthetic claim review.

The [integration guide](docs/integrations.md) covers spreadsheet intake, local API consumers, and offline healthcare contracts.

The [storage comparison](docs/storage.md) checks synthetic claim persistence and source-filtered vector search against local database engines.

The [model provider guide](docs/providers.md) describes the local runtimes and the synthetic cloud smoke command.

The [operations guide](docs/operations.md) covers container deployment, readiness and metrics, and the local load and recovery checks.

The [short explainer](docs/creative.md) illustrates cited answers, abstention, and human approval for synthetic fixture lookup.

## Evaluation

The final local holdout found reference pages for 14 of 15 questions with hybrid retrieval. The evaluated TF-IDF/Qwen answer path produced one supported answer and missed its citation-precision gate. The approval-gated synthetic workflow matched all five held-out tool tasks. [Methods, results, and limitations](docs/evaluation.md)

## Tests

The foundation defines the contracts, data policy, threat model, and reduced dependency profile. Run the dependency-free contract tests from the repository root with Python 3.12.

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

The Python helpers check basic payload structure. The JSON Schemas define the full wire contracts.

## Corpus

See the [corpus and evaluation guide](docs/corpus.md) for pinned sources, the extraction audit, and synthetic claim fixtures.

## Retrieval

The [retrieval guide](docs/retrieval.md) covers local passage search, evidence packets, and the development benchmark.

## Grounded policy answers

The [answer service guide](docs/grounded-answers.md) covers the local API, cited excerpts, abstention, and evaluation.

## Review agent

The [review agent guide](docs/review-agent.md) covers human approval, synthetic claim checks, policy lookup, and redacted traces. The [framework comparison](docs/agent-frameworks.md) reports local parity, failure behavior, and resource measurements.

## Model lab

The [synthetic fixture model lab](docs/model-lab.md) covers data mining, clustering, workload forecasting, and document-vision experiments using four local ML stacks.

The [FLAN-T5 fine-tuning lab](docs/finetune.md) compares prompt-only and fine-tuned routing on held-out synthetic claims.

## Runtime

- Python 3.12
- FastAPI and Pydantic for typed service contracts
- scikit-learn for a transparent retrieval baseline
- one measured local embedding and instruction model
- SQLite, JSON manifests, and JSONL traces
- Streamlit for a lightweight local interface

All results must report actual data scale, model identity, environment, metrics, limitations, and failures.
