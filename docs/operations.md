# Local operations

The deployment runs the existing Transformers policy runtime and approval-gated review API in one non-root process. Mount the checksum-verified model and PDFs read-only. The container image carries the service code, manifests, and synthetic fixtures.

## API behavior

| Endpoint | Operation |
| --- | --- |
| `GET /health` | Reports whether the HTTP process is alive. |
| `GET /ready` | Loads the model and checks SQLite. An unavailable dependency returns 503. |
| `POST /answer` | Routes unscoped queries to abstention before inference. Scoped results use a 60-second, 128-entry in-process cache. |
| `POST /answers/batch` | Accepts `{"queries": ["…"]}` with up to eight questions. Preserves order and computes each distinct question once. |
| `GET /metrics` | Exposes Prometheus-format request counts, durations, cache and model events, process CPU seconds, and peak RSS. |

An in-process lock prevents duplicate inference for concurrent misses on the same key. The batch evaluates unique questions sequentially and stores their summaries in a single transaction. A failed batch records one error summary. Model load and inference failures return controlled 503 responses. A subsequent readiness request or eligible cache miss can load the model again. Once the model is loaded, readiness can respond during inference.

The HTTP body limit is 16 KiB. Request traces record a generated request ID, route label, status, and duration in `api-requests.jsonl` beside SQLite. Supported routes use fixed labels and caller-supplied IDs are ignored. A single policy request uses that same ID for its SQLite summary. The review agent retains its separate redacted trace.

## Docker Compose

From the repository root with verified assets already fetched:

```sh
docker compose --project-name careops-ops13 -f deploy/compose.yaml up --build --detach --wait
curl -fsS http://127.0.0.1:8800/ready
curl -fsS http://127.0.0.1:8800/metrics
```

The API listens on host loopback. Compose stores run history and trace files in a named volume. The image uses CPU PyTorch and mounts temporary space for runtime caches. If the assets are missing, fetch them using the [application guide](application.md).

## Local Kubernetes

The local k3d configuration mounts model and PDF directories into a K3s node. The deployment requests one replica, two CPUs at its limit, and a 4 GiB memory ceiling. It uses named HTTP probes, a ClusterIP service, and ephemeral run storage within each pod. The read-only node asset mounts use hostPath in this local profile and trigger the restricted Pod Security warning.

```sh
kube=$(mktemp)
node_image=rancher/k3s:v1.35.5-k3s1@sha256:2074403abe1bded11ef3dde09d457e13be8e0b64c218b1c4f8269b4565cfbc65
docker pull "$node_image"
k3d cluster create careops-ops13 --no-lb --api-port 127.0.0.1:6551 \
  --image "$node_image" \
  --volume "$PWD/models/qwen2.5-0.5b-instruct:/mnt/careops/model:ro@server:*" \
  --volume "$PWD/data/raw:/mnt/careops/pdfs:ro@server:*" \
  --k3s-arg '--disable=traefik,servicelb,metrics-server@server:*' \
  --kubeconfig-update-default=false --kubeconfig-switch-context=false --timeout 240s
k3d kubeconfig get careops-ops13 > "$kube"
k3d image import careops-ops13:local --cluster careops-ops13
kubectl --kubeconfig "$kube" apply -f deploy/kubernetes.yaml
kubectl --kubeconfig "$kube" -n careops rollout status deployment/careops
kubectl --kubeconfig "$kube" -n careops port-forward --address 127.0.0.1 svc/careops 8802:8000
```

## Verify the deployment

Start from fresh run storage for each benchmark. Run the same public development fixture on port 8800 for Docker or port 8802 for Kubernetes.

```sh
python3 -m pip install -e '.[rag,agent,dev]'
python3 deploy/benchmark.py --target docker --port 8800
python3 deploy/benchmark.py --target kubernetes --port 8802
python3 deploy/check-recovery.py
bash deploy/check-image.sh careops-ops13:local
```

The workflow builds and tests the service and web client, smoke-tests the image, and packages the image and checksum as a delivery artifact. The local verification covers the commands and workflow validation. Hosted workflow execution and cloud deployment require a separate run.

## Measured results

| Target | Workload | Requests | p50 | p95 | Requests/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Docker | Uncached, warm model | 4 | 15,970.18 ms | 23,944.02 ms | 0.17 |
| Docker | Cached repeats | 12 | 3.71 ms | 49.78 ms | 238.92 |
| Docker | Warm batch of 8 | 1 | 29.45 ms | 29.45 ms | 33.51 |
| Kubernetes | Uncached, warm model | 4 | 14,419.96 ms | 26,944.81 ms | 0.15 |
| Kubernetes | Cached repeats | 12 | 6.65 ms | 22.11 ms | 390.69 |
| Kubernetes | Warm batch of 8 | 1 | 7.98 ms | 7.98 ms | 122.82 |

Both profiles made four initial invocations of the model-backed policy path. Each then handled twelve repeat requests from cache. The eight-question batch preserved response order. It reused four cached results and routed one distinct out-of-scope question to abstention. Each profile recorded 21 unique run summaries.

The Docker process returned 503 with missing weights and returned a cited answer after the weights were restored in the same container. Docker Compose preserved the 21 summaries across a restart. Both profiles rejected oversized bodies and malformed payloads and wrote traces with fixed route labels.

These are single-machine smoke measurements with four client workers. Uncached requests queue behind serialized CPU inference. CPU time, peak process RSS, and model-backed invocation counts are resource proxies. The measured times describe the local test profiles and do not estimate cloud pricing or production throughput. [Numerical results](../evals/results/operations.json)
