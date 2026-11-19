# Model providers

The application uses the local Transformers runtime. The provider lab compares it with loopback Ollama using the same retrieval, scope checks, selection and verification prompts, and citation validation.

The Ollama adapter requests a constrained JSON choice. It accepts one allowed label and leaves the policy workflow to abstain on invalid choices. Transport failures return a redacted error code. The adapter uses the configured loopback port.

## Reproduce the local comparison

From the repository root on Python 3.12, fetch the verified files if they are missing.

```bash
python3 -m pip install -e '.[rag,agent,providers]'
python3 -m careops.corpus fetch
python3 -m careops.answer_eval fetch-model
```

The tested Ollama version was 0.40.16. Convert the verified Qwen weights locally with llama.cpp b6451 so both runtimes start with the same checkpoint. Use the converter and its bundled GGUF library from the same release. [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b6451) · [Ollama GGUF import](https://docs.ollama.com/import)

```bash
work=$(mktemp -d)
git clone --depth 1 --branch b6451 https://github.com/ggml-org/llama.cpp "$work/llama.cpp"
python3 -m pip install 'mistral-common==1.8.3' 'sentencepiece==0.2.1' 'pydantic==2.10.5'
HF_HUB_OFFLINE=1 python3 "$work/llama.cpp/convert_hf_to_gguf.py" \
  models/qwen2.5-0.5b-instruct --outtype bf16 --outfile "$work/qwen-bf16.gguf"
printf 'FROM %s\nPARAMETER num_ctx 4096\n' "$work/qwen-bf16.gguf" > "$work/Modelfile"
printf '%s\n' "$work"
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$work/models" OLLAMA_NO_CLOUD=1 ollama serve
```

In another terminal, set `work` to the directory printed above, create the model, and run the comparison.

```bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NO_CLOUD=1 ollama create careops-qwen -f "$work/Modelfile"
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  python3 -m careops.providers.local_eval > "$work/local-providers.json"
```

`CAREOPS_OLLAMA_PORT` and `CAREOPS_OLLAMA_MODEL` select a different loopback port or imported model. The application and the original grounding and agent comparisons use Transformers.

## Development results

| Runtime | Answered | On reference page | Supported on excerpt review | Negative abstentions | p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformers CPU | 3/15 | 3/3 | 3/3 | 5/5 | 1,370 ms |
| Ollama Metal | 10/15 | 7/10 | 6/10 | 5/5 | 522 ms |

Ollama used BF16 GGUF with SHA-256 `2deba1b3b39d7afc32d8868c9c4bb3c5ce3a3d3a69ece759440d18a2e126e61e`. The decoder and hardware settings differ across the comparison. Latency is a local observation. Excerpt review found support for D-001, D-002, D-006, D-007, D-013, and D-015. D-004, D-011, and D-014 cited unrelated requirements. D-012 cited the expected page and missed the requested sequence. Ollama’s 60% supported citation rate missed the 90% development gate, so Transformers remains the application runtime. [Measured local report](../evals/results/local-providers.json)

## Synthetic cloud smoke tests

The external adapters run through a separate command that sends only three invented policy cases. The command takes a provider and model, makes at most one request per case, and stops on a transport or quota failure. Its output records normalized choices, reported token counts, and latency. Real application questions, claim data, and retrieved passages stay on the local path.

| Provider argument | Credential environment variable |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `vertex` | `VERTEX_ACCESS_TOKEN` and `GOOGLE_CLOUD_PROJECT` |

Vertex also accepts `GOOGLE_CLOUD_LOCATION` and defaults to `global`. Use the smoke command only with a provider and model whose access and costs are already authorized.

```bash
python3 -m careops.providers.smoke gemini gemini-3.1-flash-lite
python3 -m careops.providers.smoke groq openai/gpt-oss-20b
```

Gemini 3.1 Flash-Lite and Groq GPT-OSS 20B each returned the expected choice on all three live synthetic cases. [Gemini smoke results](../evals/results/provider-gemini.json) · [Groq smoke results](../evals/results/provider-groq.json)
