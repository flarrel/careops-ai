FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 HF_HOME=/tmp/careops-hf TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 LANGSMITH_TRACING=false
WORKDIR /service
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 careops && useradd -u 10001 -g careops -M careops
RUN python -m pip install --no-cache-dir --timeout 90 --retries 2 \
    --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c 'import tomllib; from pathlib import Path; p = tomllib.loads(Path("pyproject.toml").read_text()); print(*p["project"]["optional-dependencies"]["rag"], *p["project"]["optional-dependencies"]["agent"], sep="\n")' > /tmp/runtime-requirements.txt \
    && python -m pip install --timeout 90 -r /tmp/runtime-requirements.txt \
    && rm /tmp/runtime-requirements.txt \
    && mkdir -p data/runtime data/raw models/qwen2.5-0.5b-instruct \
    && chown 10001:10001 data/runtime
COPY src ./src
COPY data/manifests/ ./data/manifests/
COPY data/samples/ ./data/samples/
ENV PYTHONPATH=/service/src
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=8s --start-period=60s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/ready', timeout=7)"]
CMD ["python", "-m", "uvicorn", "careops.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
