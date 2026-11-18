import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

APP_AVAILABLE = all(importlib.util.find_spec(dep) for dep in ("fastapi", "httpx", "langgraph"))


@unittest.skipUnless(APP_AVAILABLE, "Application tests require rag, agent and app extras")
class ApplicationTests(unittest.TestCase):
    def setUp(self):
        from careops.api import create_app
        from careops.answer import abstain

        class Answerer:
            def answer(self, query):
                return abstain("no_evidence")

        directory = self.enterContext(TemporaryDirectory())
        self.path = Path(directory) / "runs.sqlite3"
        self.now = 0
        self.app = create_app(Answerer, history_path=self.path, clock=lambda: self.now)

    def test_claim_decision_private_and_history_persisted(self):
        from fastapi.testclient import TestClient

        with TestClient(self.app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            self.assertEqual(client.get("/runs").json(), [])
            option = {"claim_id": "SYN-D-001"}
            pending = client.post("/reviews", json=option)
            self.assertEqual(pending.status_code, 202)
            self.assertEqual(pending.json().keys(), {"run_id", "status"})
            self.assertEqual(client.get("/runs").json(), [])
            url = f'/reviews/{pending.json()["run_id"]}/decision'
            self.assertEqual(client.post(url, json={"approved": "true"}).status_code, 422)
            result = client.post(url, json={"approved": True})
            self.assertEqual(result.json()["lookup"]["claim_id"], "SYN-D-001")
            self.assertEqual(result.json()["result"]["status"], "escalate")
            self.assertEqual(client.post(url, json={"approved": True}).status_code, 404)

            another = client.post("/reviews", json=option).json()["run_id"]
            refused = client.post(f"/reviews/{another}/decision", json={"approved": False})
            self.assertIsNone(refused.json()["lookup"])
            self.assertEqual(refused.json()["result"]["tool_calls"], [])
            response = client.post("/answer", json={"query": "private text"})
            self.assertEqual(response.headers["cache-control"], "no-store")

        with TestClient(self.app) as client:
            runs = client.get("/runs")
            self.assertEqual([r["status"] for r in runs.json()], ["abstain", "abstain", "escalate"])
            self.assertEqual(client.get("/runs?limit=1").json(), runs.json()[:1])
            self.assertNotIn("private text", runs.text)
            self.assertNotIn("SYN-D-001", runs.text)
            self.assertNotIn(b"private text", self.path.read_bytes())
            self.assertNotIn(b"SYN-D-001", self.path.read_bytes())

    def test_expired_and_invalid_inputs_have_no_payload(self):
        from fastapi.testclient import TestClient
        from careops.api import APPROVAL_TTL, MAX_PENDING

        with TestClient(self.app) as client:
            bad = client.post("/reviews", json={"claim_id": "sensitive-value"})
            self.assertEqual(bad.status_code, 422)
            self.assertNotIn("sensitive-value", bad.text)
            payload = {"claim_id": "SYN-D-001"}
            pending = [client.post("/reviews", json=payload) for _ in range(MAX_PENDING)]
            self.assertTrue(all(item.status_code == 202 for item in pending))
            self.assertEqual(client.post("/reviews", json=payload).status_code, 429)
            self.now += APPROVAL_TTL
            url = f'/reviews/{pending[0].json()["run_id"]}/decision'
            self.assertEqual(client.post(url, json={"approved": True}).status_code, 404)
            self.assertEqual(client.post("/reviews", json=payload).status_code, 202)
            self.assertEqual(client.get("/runs").json(), [])
