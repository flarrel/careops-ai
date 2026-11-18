# Grounded policy answers

The local answer service looks up passages in the five development PDFs with TF-IDF. It asks for a regulation, year, article, or appendix reference when the question lacks one, then checks that scope against the source edition and the retrieved passage location. Appendix checks start at an extracted standalone `LAMPIRAN` heading. The Qwen instruction model selects one candidate, then makes a second pass to check whether that passage fully answers the question. The response contains the exact selected excerpt and a citation resolved from the indexed passage and the source manifest. A rejected or unresolved selection produces an abstention. Five candidates at most reach the model after filtering the first 20 TF-IDF hits. The service can return a single supporting passage.

The model processes the question and the retrieved text as JSON data. Chat delimiters are escaped, selection is restricted to listed passage labels, and the service has no action tools. The model can still misjudge relevance or miss evidence across passages. Citations expose the source edition and physical PDF page so the reader can check the excerpt.

The [model manifest](../data/manifests/generation.json) pins five files, including the Apache-2.0 Qwen2.5-0.5B-Instruct weights. The files total 995,137,598 bytes and are verified before loading. [Pinned model revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2)

## Run locally

Run from the repository root with Python 3.12.

```sh
python3 -m pip install -e '.[rag,dev]'
python3 -m careops.corpus fetch
python3 -m careops.answer_eval fetch-model
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python3 -m uvicorn careops.api:app \
  --host 127.0.0.1 --port 8000
```

```sh
curl -sS http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"Siapa pengguna pedoman INA-CBG untuk metode pembayaran klaim menurut Permenkes 26/2021?"}'
```

`POST /answer` accepts a question of 1–500 characters. It returns an `answer` or `abstain` status. Cited answers include the document title, URL, passage ID, and page range. An abstention includes an Indonesian message, a reason code, and an empty citation list. The server loads the corpus and model on the first policy request and reuses them. Inference uses the local model files after the explicit fetch.

## Development evaluation

Reproduce the same-model comparison and apply the saved judgments:

```sh
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python3 -m careops.answer_eval benchmark \
  > /tmp/careops-answer-predictions.json
python3 -m careops.answer_eval summarize /tmp/careops-answer-predictions.json \
  evals/grounding-review.json
```

The first command produces a local file with the model outputs. The public [review](../evals/grounding-review.json) stores output hashes, judgments, and notes. A different output needs a new review. The results cover 15 policy questions, three unanswerable questions, and two instruction-boundary questions. Claim tool cases belong to a separate evaluation.

The reviewer compared each answer with the authored expected facts. A correct answer states all requested facts. Citation precision requires a supporting excerpt on an expected source page. Unsupported answer rate counts wrong, incomplete, or unrelated responses among answers returned. Abstention includes a clear refusal or admission of insufficient information in the no-retrieval comparator. The unsupported answer metric is response-level and does not count individual factual claims. One local review and a small development set limit the inference.

The [development results](../evals/results/grounded-development.json) show that selective abstention improved evidence quality. The three emitted excerpts answered their questions and cited the expected pages. The service abstained on 12 of 15 answerable cases, so the measured citation precision rests on three citations.

| Measure | Grounded excerpts | No retrieval |
|---|---:|---:|
| Answerable accuracy | 3/15 | 0/15 |
| Answerable coverage | 3/15 | 14/15 |
| Relevant citation precision | 3/3 | — |
| Unanswerable abstention | 3/3 | 0/3 |
| Instruction-boundary abstention | 2/2 | 1/2 |
| Unsupported answers among returned answers | 0/3 | 18/18 |
| Warm p50 / p95 | 1.08 / 1.84 s | 1.18 / 1.61 s |

The development precision and unanswerable-abstention targets were both 90%. Both passed on these limited denominators. The three unanswerable and two instruction-boundary questions lacked a policy reference, so their outcomes exercised the request scope check. Two additional manual API probes supplied real regulation references while requesting an approval action and a facility-specific fee. Both abstained. Broader in-scope instruction-boundary testing remains open. The no-retrieval baseline followed the zero-tariff instruction in one case and supplied an invented payment in another. The grounded service abstained even on some correctly retrieved sources, including the two tariff categories in D-006. D-003 required evidence across editions, which a one-passage response cannot reliably handle. Improving answer coverage would require better passage verification and support for combining citations.

The benchmark ran on an Apple M3 Pro with four CPU threads. Peak RSS for both paths in one process was about 3.40 GiB. Warm latency includes retrieval and model steps, and excludes startup. The held-out set stayed sealed.
