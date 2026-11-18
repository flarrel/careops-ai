# Foundation feasibility record

Recorded during checkpoint 1 on 2026-11-18 (Asia/Jakarta).

| Item | Observation | Decision |
|---|---|---|
| Host | Apple M3 Pro (arm64), 36 GB RAM, macOS 14.3 | PyTorch reports an MPS-enabled build with MPS unavailable in the verification session. Use CPU as the initial baseline. |
| Python | Python 3.12.4 | Use Python 3.12 and a project virtual environment. |
| Disk | Approximately 123 GiB free | Small models and fixtures are acceptable; large downloads remain gated. |
| Docker | Docker 29.9.2 and Compose v5.6.1 present; daemon probe failed | Do not depend on Docker in checkpoint 1. Recheck before container work. |
| Local model runner | Ollama unavailable | Select a small local model only after a later measured feasibility check. |
| Frontend/toolchains | Node, npm, PHP, Flutter, Blender, Tesseract unavailable | Keep later modules gated; no fake compatibility evidence. |
| Available baseline tools | Python, Git, SQLite, Java, Go, Rust, kubectl, FFmpeg | None is added merely for keywords; use only where a real checkpoint needs it. |
| External accounts | No provider credentials or accounts inspected | No external account action is required for the reduced profile. |
| Source licenses | Corpus and optional reference datasets are pending selection | Check and record use and redistribution terms for each source before retaining files. |

## Local generation candidate

The initial local candidate is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2), licensed under Apache-2.0. The [published weights pointer](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/commit/eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2) records `model.safetensors` at 988,097,824 bytes (about 0.92 GiB) with SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`. The weight file fits the observed disk headroom. Runtime memory, latency, and Indonesian policy grounding remain to be measured before adoption.

Foundation verification used local tools and upstream metadata. Model weights, datasets, and database images were kept out of the foundation.
