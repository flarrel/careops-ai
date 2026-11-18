# Architecture

The first maintained path is a single local application with explicit contracts:

1. A manifest records source identity, URL, retrieval date, license note, version, and checksum.
2. Ingestion extracts text and creates deterministic chunks with source locations.
3. Retrieval exposes a transparent lexical baseline and a measured dense index.
4. The answer layer receives only a typed evidence packet and must cite its source IDs or abstain.
5. A bounded agent may call only typed read-only policy and synthetic-claim tools.
6. FastAPI owns the workflow; Streamlit renders responses and never reimplements inference.
7. SQLite stores small run records and JSONL stores redacted traces.

Comparison modules reuse these contracts. They do not create parallel products or silently replace the maintained path.
