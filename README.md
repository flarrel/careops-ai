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

The project will use public, provenance-recorded policy material and project-owned synthetic records. It is not medical advice, a clinical decision system, a live claims processor, or a substitute for professional review.

## Foundation status

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

The [review agent guide](docs/review-agent.md) covers human approval, synthetic claim checks, policy lookup, and redacted traces.

## Planned runtime

- Python 3.12
- FastAPI and Pydantic for typed service contracts
- scikit-learn for a transparent retrieval baseline
- one measured local embedding and instruction model
- SQLite, JSON manifests, and JSONL traces
- Streamlit for a lightweight local interface

All results must report actual data scale, model identity, environment, metrics, limitations, and failures.
