# Final local evaluation

CareOps AI retrieves excerpts from Indonesian public policy and reads synthetic claim fixtures through an approval-gated workflow. The final holdout exposed a gap in policy answers. The best retrieval configuration found the reference page for 14 of 15 questions, while the frozen TF-IDF/Qwen answer pipeline returned only one supported answer. The fixture workflow matched all five expected outcomes.

## Corpus and methods

The corpus contains 10 checksum-pinned official PDFs from four document families. The extraction audit inspected 340 pages and flagged 88 with fewer than 100 characters, including empty pages. Normalization preserves page numbers and character offsets. Article-aware chunking produced 909 passages across the full corpus. The claim fixtures are project-authored and synthetic. [Corpus and provenance](corpus.md) · [Data policy](data-policy.md)

Development and final question families, fixture templates, IDs, and question text were checked for overlap by the seal verifier. The final inputs contain 15 source-backed questions, three unanswerable questions, two instruction-boundary cases, and five tool tasks. The file hashes matched the manifest originally recorded in `evals/sealed.sha256`. The local evaluation indexed all ten approved source documents because the maintained development index omits the two held-out document families. The comparison kept model weights, prompt, scope rules, and top-k fixed. Fixture labels were used for scoring, outside the model calls.

Retrieval compares word/bigram TF-IDF, multilingual E5 dense cosine search, and reciprocal-rank fusion. Page Recall@5 measures how many reference pages appeared in the first five passage hits. MRR@5 rewards finding the first reference page earlier. Source hit rate gives credit for the correct document even if the selected page is wrong. The answer comparison uses the same local Qwen model, either choosing and verifying an excerpt from the TF-IDF shortlist or generating without retrieval. Manual grading checked whether the result answered the authored question and whether the cited passage supported it. The tool comparator uses a direct policy answer with no claim lookup. [Retrieval design](retrieval.md) · [Answer design](grounded-answers.md) · [Review agent](review-agent.md)

## Retrieval

| Method | Page Recall@5 | MRR@5 | Source hit rate | p95 | Numeric index payload |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 0.7333 | 0.5222 | 0.9333 | 0.87 ms | 1,184,168 bytes |
| E5 dense | 0.6000 | 0.4667 | 1.0000 | 17.65 ms | 1,396,224 bytes |
| Reciprocal-rank fusion | 0.9333 | 0.5822 | 1.0000 | 17.73 ms | 2,580,392 bytes |

Standalone dense retrieval missed the improvement gate. Fusion improved both ranking metrics. Its stronger page recall establishes retrieval quality on this set. The answer experiment used TF-IDF, so the evaluation supplies no generation quality measurement for a fusion-backed answer path. Numeric payloads omit model weights, strings and Python overhead.

## Answers and error analysis

| Configuration | Supported answers on 15 | Answered on 15 | Unanswerable abstentions | Instruction-boundary abstentions | Supported citations |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Qwen, full approved corpus | 1 | 4 | 3/3 | 2/2 | 1/4 |
| Qwen without retrieval | 0 | 15 | 0/3 | 0/2 | — |

The cited path missed the 90% citation-precision gate. Three excerpts were unsupported. One used the correct PDF with the wrong page, another used a transitional provision unrelated to the requested effective date, and a third was on the reference page but addressed transfer instead of ownership. A matching page is therefore insufficient to score an answer as supported. The report separates shortlist misses, nearby passages removed by scope filtering, and abstentions despite an eligible reference page. Ten questions abstained despite an eligible reference page, and one abstained after scope filtering excluded the expected page. Some questions require evidence spread across clauses while the current selector returns a single excerpt.

The unchanged development-index application configuration abstained on 14 of 15 held-out policy questions and returned one excerpt from a different document family. The question gave an article but omitted a document edition. This shows that article-only scope can select the same-numbered provision in an unrelated document. The five negative held-out questions were blocked by the scope check. The observed abstention rate describes those fixed cases. Scoped instruction attacks are a separate coverage gap.

## Tools and operations

The workflow matched five of five synthetic tasks, all five evaluated fields, and three of three expected escalations. All five evaluated results passed the schema, and all 28 trace events validated. The direct-answer comparator completed zero tasks. The approval pause and explicit denial blocked lookup, malformed input was rejected, and the injected timeout escalated after one retry. No unauthorized tool attempts or synthetic fixture identifiers appeared in the tested traces. The fixture result is an engineering check of status and field handling. It carries no clinical or payment authority.

The runtime benchmark used the FastAPI application with an in-process HTTP client, the maintained development configuration, and a fresh temporary SQLite database. On a MacBook Pro (Mac15,7) with an Apple M3 Pro, 36 GB RAM, macOS 14.3, and four Torch threads, the named run measured p50/p95 of 1,719/2,080 ms on the first eight policy requests and 1.02/1.58 ms on eight repeats. A batch containing a repeated request preserved order. The database recorded the expected 19 summaries with unique IDs. The benchmark observed 422 for malformed input, 413 for an oversized body, and recovery after an injected model-load failure. The full evaluation process peaked at approximately 4.50 GB RSS, including both models, index construction, and the runtime check. This is a local process measurement. Previous container and single-node orchestration measurements are recorded separately. [Operations guide](operations.md) · [Build record](../evals/results/final.json)

## Reproduce and inspect

Install `.[rag,agent,dev]` on Python 3.12 and prepare the hash-pinned policy PDFs and model weights as described in the corpus, retrieval, and answer guides. With those inputs cached, the commands below use CPU inference offline. `HOLDOUT_DIR` is a private directory with the original manifest and JSONL files. `JUDGMENTS` is the private manual-review file with judgments bound to the exact prediction digests. Case-level predictions remain outside the repository.

```sh
work=$(mktemp -d)
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python3 -m careops.corpus audit > "$work/corpus.json"
python3 -m careops.final_eval "$HOLDOUT_DIR" --output "$work/predictions.json" predict \
  --machine "Machine model, CPU and memory"
python3 -m careops.final_eval "$HOLDOUT_DIR" --output "$work/final.json" summarize \
  "$work/predictions.json" "$JUDGMENTS"
python3 -B -m unittest discover -s tests -v
```

Two cold-process runs reproduced all 60 answer-path outputs and all 45 ranked queries. Pass `--replay` to `summarize` with a second private prediction file to check the same outputs.

Use the development corpus and public fixtures for open reproduction of the application and its benchmarks. The final report includes only aggregate metrics. Replaying the private benchmark requires the original holdout and review artifacts. No clinical evaluation, independent grading, production traffic, hosted CI execution, or billing validation was performed in this checkpoint. The development results, fine-tuning holdout, and final policy split describe different tasks. [Application guide](application.md) · [Fine-tuning comparison](finetune.md) · [Model lab](model-lab.md)
