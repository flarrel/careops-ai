import importlib.util
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

OPS_AVAILABLE = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))
Q1 = "Apa isi Pasal 1 Permenkes 1/2022?"
Q2 = "Apa isi Pasal 2 Permenkes 1/2022?"


@unittest.skipUnless(OPS_AVAILABLE, "Operations tests require FastAPI and HTTPX")
class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(self.enterContext(TemporaryDirectory())) / "runs.sqlite3"
        self.calls = []
        self.loads = 0
        self.now = 1

    def create_client(self, loader=None):
        from fastapi.testclient import TestClient

        from careops.api import create_app

        return self.enterContext(TestClient(create_app(
            loader or self.load_fake, history_path=self.path, clock=lambda: self.now,
        )))

    def load_fake(self):
        from careops.answer import abstain

        self.loads += 1

        class Backend:
            def answer(_, query):
                self.calls.append(query)
                return abstain("no_evidence")

        return Backend()

    def test_cache_scope_and_expiration(self):
        from careops.operations import CACHE_TTL

        client = self.create_client()
        self.assertEqual(client.post("/answer", json={"query": "No scope"}).json()["reason"],
                         "scope_required")
        self.assertEqual(self.loads, 0)
        self.assertEqual(client.get("/ready").json(), {"status": "ready"})
        for _ in range(3):
            self.assertEqual(client.post("/answer", json={"query": Q1}).status_code, 200)
        self.assertEqual(self.calls, [Q1])
        self.now += CACHE_TTL
        self.assertEqual(client.post("/answer", json={"query": Q1}).status_code, 200)
        self.assertEqual((self.loads, self.calls), (1, [Q1, Q1]))
        metrics = client.get("/metrics").text
        self.assertIn('careops_policy_events_total{event="cache_hit"} 2', metrics)
        self.assertIn('careops_policy_events_total{event="scope_abstain"} 1', metrics)

    def test_batch_preserves_order_and_single_summary_per_unique_query(self):
        client = self.create_client()
        response = client.post("/answers/batch", json={"queries": [Q1, "No scope", Q1, Q2]})
        self.assertEqual([item["reason"] for item in response.json()],
                         ["no_evidence", "scope_required", "no_evidence", "no_evidence"])
        self.assertEqual(self.calls, [Q1, Q2])
        ids = [row["run_id"] for row in client.get("/runs").json()]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(len(ids), 3)
        self.assertEqual(client.post("/answers/batch", json={"queries": [Q1] * 9}).status_code,
                         422)

    def test_readiness_recovers_and_failed_batch_keeps_no_partial_summaries(self):
        from careops.answer import abstain

        def loader():
            self.loads += 1
            if self.loads == 1:
                raise OSError("private model path")

            class Backend:
                def answer(_, query):
                    if query == Q2 and self.loads == 2:
                        raise RuntimeError("private inference details")
                    return abstain("no_evidence")

            return Backend()

        client = self.create_client(loader)
        self.assertEqual(client.get("/health").status_code, 200)
        cold = client.get("/ready")
        self.assertEqual(cold.status_code, 503)
        self.assertNotIn("private", cold.text)
        self.assertEqual(client.get("/ready").status_code, 200)
        failed = client.post("/answers/batch", json={"queries": [Q1, Q2]})
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("private", failed.text)
        self.assertEqual([row["status"] for row in client.get("/runs").json()], ["error"])
        self.assertEqual(client.post("/answer", json={"query": Q2}).status_code, 200)
        self.assertEqual(self.loads, 3)

    def test_loaded_readiness_is_independent_of_in_flight_inference(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        from careops.answer import abstain

        entered, release = Event(), Event()

        class Backend:
            def answer(self, _query):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release inference")
                return abstain("no_evidence")

        client = self.create_client(Backend)
        self.assertEqual(client.get("/ready").status_code, 200)
        with ThreadPoolExecutor(max_workers=2) as pool:
            active = pool.submit(client.post, "/answer", json={"query": Q1})
            try:
                self.assertTrue(entered.wait(2))
                probe = pool.submit(client.get, "/ready")
                self.assertEqual(probe.result(timeout=2).status_code, 200)
            finally:
                release.set()
            self.assertEqual(active.result(timeout=2).status_code, 200)

    def test_request_bounds_and_traces_omit_inputs(self):
        client = self.create_client()
        response = client.post("/answer", content=b"private" * 6_000,
                               headers={"X-Request-ID": "private-id"})
        self.assertEqual(response.status_code, 413)
        self.assertRegex(response.headers["x-request-id"], re.compile("^[a-f0-9]{32}$"))
        self.assertNotIn("private", response.text)
        self.assertEqual(client.post("/answer", json={"query": "private-input x", "extra": True})
                         .status_code, 422)
        client.get("/private-path")
        lines = self.path.with_name("api-requests.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["status"] for line in lines], [413, 422, 404])
        self.assertTrue(all("private" not in line for line in lines))
        self.assertEqual(self.loads, 0)
