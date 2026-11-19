#!/usr/bin/env bash
set -euo pipefail

image=${1:?Pass a local image tag}
container=$(docker run --detach --read-only --cap-drop ALL --security-opt no-new-privileges \
    --tmpfs /tmp:mode=1777 --tmpfs /service/data/runtime:uid=10001,gid=10001,mode=0700 "$image")
trap 'docker rm -f "$container" >/dev/null' EXIT
docker exec -i "$container" python - <<'PY'
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

deadline = monotonic() + 45
while True:
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=2) as reply:
            assert reply.status == 200
        break
    except URLError:
        if monotonic() >= deadline:
            raise
        sleep(0.25)
try:
    urlopen("http://127.0.0.1:8000/ready", timeout=10)
except HTTPError as error:
    assert error.code == 503
else:
    raise RuntimeError("An image with no mounted PDFs or weights became ready")
print("Image liveness and missing-asset readiness passed")
PY
