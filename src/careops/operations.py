"""Bounded inference routing, memoization, and redacted HTTP telemetry."""

import logging
import platform
import resource
from collections import Counter, OrderedDict
from threading import Lock
from time import monotonic, perf_counter
from uuid import uuid4

from starlette.responses import JSONResponse

from careops.answer import abstain, requested_scope

MAX_BODY_BYTES = 16 * 1024
BODY_METHODS = ("POST", "PUT", "PATCH")
CACHE_SIZE = 128
CACHE_TTL = 60
ROUTES = {
    ("POST", "/answer"): "answer",
    ("POST", "/answers/batch"): "answer_batch",
    ("POST", "/reviews"): "review_start",
    ("GET", "/runs"): "history",
}
PROBES = ("/health", "/ready", "/metrics")


class Operations:
    def __init__(self):
        self.lock = Lock()
        self.requests = Counter()
        self.seconds = Counter()
        self.policy = Counter()

    def event(self, name):
        with self.lock:
            self.policy[name] += 1

    def observe(self, route, status, seconds):
        with self.lock:
            self.requests[route, status] += 1
            self.seconds[route] += seconds

    def exposition(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = usage.ru_maxrss
        peak = int(rss if platform.system() == "Darwin" else rss * 1024)
        cpu = usage.ru_utime + usage.ru_stime
        with self.lock:
            lines = ["# TYPE careops_http_requests_total counter"]
            lines.extend(
                f'careops_http_requests_total{{route="{route}",status="{status}"}} {count}'
                for (route, status), count in sorted(self.requests.items())
            )
            lines.append("# TYPE careops_http_request_seconds_total counter")
            lines.extend(f'careops_http_request_seconds_total{{route="{route}"}} {seconds:.9f}'
                         for route, seconds in sorted(self.seconds.items()))
            lines.append("# TYPE careops_policy_events_total counter")
            lines.extend(f'careops_policy_events_total{{event="{name}"}} {count}'
                         for name, count in sorted(self.policy.items()))
        return "\n".join([
            *lines, "# TYPE careops_process_peak_rss_bytes gauge",
            f"careops_process_peak_rss_bytes {peak}", "# TYPE careops_process_cpu_seconds_total counter",
            f"careops_process_cpu_seconds_total {cpu:.6f}",
        ]) + "\n"


class PolicyRuntime:
    def __init__(self, load, validate, operations, *, clock=monotonic):
        self.load = load
        self.validate = validate
        self.operations = operations
        self.clock = clock
        self.service = None
        self.fill_lock = Lock()
        self.cache_lock = Lock()
        self.cache = OrderedDict()

    def _load(self):
        if self.service is None:
            self.operations.event("model_load")
            try:
                self.service = self.load()
            except (OSError, RuntimeError, ValueError):
                self.operations.event("model_load_error")
                raise

    def warm(self):
        if self.service is None:
            with self.fill_lock:
                self._load()

    def cached(self, query):
        with self.cache_lock:
            item = self.cache.get(query)
            if item is None:
                return None
            expires, result = item
            if expires <= self.clock():
                del self.cache[query]
                return None
            self.cache.move_to_end(query)
            return result

    def answer(self, query):
        if requested_scope(query) is None:
            self.operations.event("scope_abstain")
            return abstain("scope_required")
        result = self.cached(query)
        if result is not None:
            self.operations.event("cache_hit")
            return result
        with self.fill_lock:
            result = self.cached(query)
            if result is not None:
                self.operations.event("cache_hit")
                return result
            self.operations.event("cache_miss")
            self._load()
            self.operations.event("model_answer")
            try:
                result = self.validate(self.service.answer(query))
            except (OSError, RuntimeError, ValueError):
                self.operations.event("model_answer_error")
                self.service = None
                raise
            with self.cache_lock:
                self.cache[query] = (self.clock() + CACHE_TTL, result)
                if len(self.cache) > CACHE_SIZE:
                    self.cache.popitem(last=False)
            return result


def route_label(method, path):
    if method == "POST" and path.startswith("/reviews/") and path.endswith("/decision"):
        return "review_decision"
    return ROUTES.get((method, path), "other")


class RequestOperations:
    def __init__(self, app, operations, trace):
        self.app = app
        self.operations = operations
        self.trace = trace

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method, path = scope["method"], scope["path"]
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = perf_counter()

        async def response(message):
            if message["type"] == "http.response.start":
                message["headers"] = [
                    (key, value) for key, value in message["headers"]
                    if key not in (b"cache-control", b"x-request-id")
                ] + [(b"cache-control", b"no-store"), (b"x-request-id", request_id.encode())]
                if path not in PROBES:
                    route = route_label(method, path)
                    seconds = perf_counter() - started
                    self.operations.observe(route, message["status"], seconds)
                    try:
                        self.trace({"request_id": request_id, "route": route,
                                    "status": message["status"], "duration_ms": round(seconds * 1000, 3)})
                    except OSError:
                        self.operations.event("request_trace_error")
                        logging.getLogger(__name__).warning("Could not append the request trace")
            await send(message)

        if method in BODY_METHODS:
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                block = message.get("body", b"")
                if len(body) + len(block) > MAX_BODY_BYTES:
                    error = JSONResponse({"detail": "Request body exceeds 16 KiB."}, status_code=413)
                    await error(scope, receive, response)
                    return
                body.extend(block)
                if not message.get("more_body"):
                    break

            async def replay():
                nonlocal body
                if body is not None:
                    initial, body = bytes(body), None
                    return {"type": "http.request", "body": initial, "more_body": False}
                return await receive()

            await self.app(scope, replay, response)
        else:
            await self.app(scope, receive, response)
