# Storage and vector comparison

This lab runs disposable backends against the same CareOps development data. The application uses SQLite for its regular run history.

## Claim records

Each engine stores all 20 synthetic development claims. The second write replays the batch with one changed service and amount. A fresh connection checks every row, confirms that the replay kept the same count, and checks that an absent ID returns no row. The range query filters by service and minimum IDR amount, sorts by amount and claim ID descending, and returns up to three rows. A Python reference checks the order and the empty-range case.

- SQLite, PostgreSQL, and MySQL use parameterized upserts and a composite index for the range query.
- MongoDB uses replacement upserts, a compound index, and a sorted query with a limit.
- Redis writes the record hash and amount-sorted set in one transaction. It applies the amount range on the server and filters service type over that small result in Python.

| Backend | Rows after replay | Range p50 | Range p95 |
| --- | ---: | ---: | ---: |
| SQLite 3.45.3 | 20 | 0.006 ms | 0.020 ms |
| PostgreSQL 16.15 | 20 | 0.305 ms | 0.576 ms |
| MySQL 8.4.13 | 20 | 0.289 ms | 0.379 ms |
| MongoDB 8.0.38 | 20 | 0.374 ms | 0.636 ms |
| Redis 8.2.3 | 20 | 0.485 ms | 1.052 ms |

## Policy vectors

The lab embeds all 441 development passages and 15 answerable development questions with pinned multilingual E5. Every engine receives identical 384-dimensional vectors. The lab stores only vectors, chunk IDs, and source IDs in the vector engines. NumPy computes the exact cosine reference. Chroma uses HNSW, Qdrant requests exact cosine search, and Weaviate uses a flat cosine index. [Embedding manifest](../data/manifests/embedding.json)

Each engine must retain 441 entries after replaying one vector. The acceptance check requires at least 95% mean top-five ID overlap with NumPy. It also checks a source-scoped query against exact results and requires an unknown source filter to return zero hits. Page recall and reciprocal rank are computed from the development reference pages. Matching NumPy preserves the embedding baseline’s retrieval gaps.

| Backend | Page recall@5 | MRR@5 | Mean overlap@5 | Search p50 | Search p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NumPy exact | 0.6333 | 0.4189 | 100% | 0.121 ms | 0.130 ms |
| Chroma 1.5.9 | 0.6333 | 0.4189 | 100% | 0.320 ms | 0.364 ms |
| Qdrant 1.17.1 | 0.6333 | 0.4189 | 100% | 0.886 ms | 1.018 ms |
| Weaviate 1.35.13 | 0.6333 | 0.4189 | 100% | 1.923 ms | 3.024 ms |

Each row engine measured 15 range queries and each vector engine measured 75 search requests. Timings include the local request or client call and exclude embedding generation, database startup, and index creation. The comparison is a small ARM64 workstation smoke benchmark. Client overhead and index configuration affect the timings. The numerical report records the dependency versions, input fingerprints, timings, and returned IDs. [Measured report](../evals/results/storage-comparison.json)

## Run locally

From the repository root on Python 3.12, use the verified local policy PDFs and model cache.

```bash
python3 -m pip install -e '.[rag,storage]'
python3 -m careops.corpus fetch
python3 -m careops.retrieval_eval fetch-model
docker compose --project-name careops-storage-lab -f storage/compose.yaml up --detach --wait
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 HF_HUB_OFFLINE=1 \
  python3 -m careops.storage.benchmark > /tmp/careops-storage.json
docker compose --project-name careops-storage-lab -f storage/compose.yaml down --volumes
```

The pinned Compose services bind to loopback. The lab creates temporary SQLite and Chroma files and uniquely named database tables, collections, or keys. `--track rows` runs the claim checks with PostgreSQL, MySQL, MongoDB, and Redis. `--track vectors` runs vector searches with Qdrant and Weaviate plus local NumPy and Chroma.

Firebase and Pinecone require separately authorized service access for live verification. Their coverage remains open.
