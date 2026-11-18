# Policy passage retrieval

The retriever operates on the checksum-pinned development PDFs. It uses the same validated native extraction as the corpus audit and skips pages with fewer than 100 normalized characters. Passages stay within a physical PDF page. Where extraction places a `Pasal` heading on its own line, the chunker also preserves that article boundary. Longer sections use windows of 100 whitespace-delimited words with 20 words of overlap. Each chunk ID includes the source, page, offsets in the normalized page, and a text fingerprint.

The evidence packet contains ranked passages, source IDs, scores, and page and character ranges. Use the source manifest to resolve the document edition and original URL. Scores reflect ranking within a method and have no calibrated confidence interpretation. Retrieval leaves answer generation and abstention decisions to the consumer.

## Run locally

With Python 3.12 and the [source PDFs](corpus.md) in the local cache:

```sh
python3 -m pip install -e '.[retrieval]'
python3 -m careops.corpus fetch
python3 -m careops.retrieval_eval fetch-model
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python3 -m careops.retrieval_eval search \
  --method reranked "Apa indikator mutu rumah sakit menurut Permenkes 30/2022?"
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python3 -m careops.retrieval_eval benchmark \
  > /tmp/careops-retrieval-development.json
```

Search defaults to TF-IDF and five hits. Available methods are `tfidf`, `dense`, `reranked`, `langchain`, and `llamaindex`. The benchmark defaults to all five. Use `--methods tfidf` to run a baseline without the model. Each CLI invocation builds its indexes in memory. Search and benchmarking use local files after the explicit fetches.

The [embedding manifest](../data/manifests/embedding.json) pins the MIT-licensed multilingual E5 small revision and all five model file hashes. The total download is about 465 MiB and stays in the ignored model cache. Transformers runs the encoder on CPU with E5 query and passage prefixes, attention-mask mean pooling, and unit normalization. [Pinned encoder revision](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3)

## Development benchmark

Five development PDFs produced 441 chunks. The [results](../evals/results/retrieval-development.json) include the 15 policy questions and their per-case provenance. Recall@5 is the mean fraction of expected source-page pairs found. MRR@5 gives credit for the rank of the first expected page. Source hit measures a match to any expected document. The tool and abstention cases are outside this retrieval benchmark.

| Method | Page Recall@5 | Page MRR@5 | Source hit@5 | p50 / p95 (ms) | Numeric index payload |
|---|---:|---:|---:|---:|---:|
| TF-IDF cosine | 0.7333 | 0.3822 | 1.0000 | 0.31 / 0.39 | 631,036 B |
| E5 cosine | 0.6333 | 0.4189 | 1.0000 | 11.95 / 23.99 | 677,376 B |
| Reciprocal-rank reranking | 0.9000 | 0.5467 | 1.0000 | 12.09 / 20.46 | 1,308,412 B |
| LangChain InMemoryVectorStore | 0.6333 | 0.4189 | 1.0000 | 14.62 / 16.16 | 1,354,752 B |
| LlamaIndex SimpleVectorStore | 0.6333 | 0.4189 | 1.0000 | 14.93 / 15.87 | 1,354,752 B |

The reranker fuses the best rank per page from each candidate list, using reciprocal rank with a constant of 60. The comparison includes page diversification as part of reranking, so the gain cannot be attributed to fusion alone. Both framework stores ran their own cosine search over the same E5 vectors and matched the native dense top five on all development questions. The numeric payload counts sparse arrays and IDF weights, or vector elements at their numeric width. It excludes text, Python object overhead, vocabulary storage, and model weights. Query timings are from an Apple M3 Pro with four CPU threads, after one warmup query per method, and exclude index construction.

Even at 100% document hit rate, the correct page can be missing. For D-004, all paths missed the page describing the physician's role. The reranker preferred later INA-CBG coding and claim guidance. For D-003, reranking found the repeal clause but missed the expected title page of the older edition. This illustrates the cost of queries that need evidence across editions. Sparse extraction, split passages, and repeated regulatory language can all limit retrieval quality.

TF-IDF remains the default pending the held-out comparison. Keep the held-out cases and labels closed during development. None of the development results establishes answer quality or policy validity.
