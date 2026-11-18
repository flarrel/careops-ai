import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from careops.storage.rows import connect, exercise, expected_ids


class StorageTests(unittest.TestCase):
    def test_sqlite_replay_preserves_rows_after_reconnect(self):
        rows = [
            {"claim_id": "SYN-D-001", "service_type": "outpatient", "service_code": "SYN-OUT-A",
             "amount_idr": 120_000, "attachments": ["invoice"]},
            {"claim_id": "SYN-D-002", "service_type": "inpatient", "service_code": "SYN-IN-A",
             "amount_idr": 230_000, "attachments": ["signed-discharge-summary"]},
        ]
        with TemporaryDirectory() as directory:
            result, timings = exercise("sqlite", Path(directory), "claim_lab_abcdef123456", rows)
        self.assertEqual(result["rows"], 2)
        self.assertTrue(result["upsert_and_reopen"])
        self.assertEqual(result["queries"][0]["claim_ids"], ["SYN-D-001", "SYN-D-002"])
        self.assertEqual(result["queries"][-1]["claim_ids"], [])
        self.assertEqual(len(timings), 15)

    def test_amount_ties_and_disposable_namespace(self):
        rows = [{"claim_id": claim, "service_type": "outpatient", "amount_idr": 100_000}
                for claim in ("SYN-D-001", "SYN-D-002")]
        self.assertEqual(expected_ids(rows, "outpatient", 100_000, 1), ["SYN-D-002"])
        with TemporaryDirectory() as directory, self.assertRaises(ValueError):
            connect("sqlite", Path(directory), "runs")


@unittest.skipUnless(
    all(importlib.util.find_spec(dep) for dep in ("httpx", "numpy", "pydantic")),
    "Vector acceptance helpers require the storage and rag extras",
)
class VectorAcceptanceTests(unittest.TestCase):
    def test_exact_search_applies_source_filters(self):
        from types import SimpleNamespace

        from numpy import array

        from careops.storage.benchmark import exact_ids

        chunks = [SimpleNamespace(chunk_id=key, source_id=source)
                  for key, source in (("a", "left"), ("b", "right"), ("c", "left"))]
        matrix = array([[1, 0], [0, 1], [1, 0]])
        query = array([1, 0])
        self.assertEqual(exact_ids(chunks, matrix, query), ["a", "c", "b"])
        self.assertEqual(exact_ids(chunks, matrix, query, "right"), ["b"])
        self.assertEqual(exact_ids(chunks, matrix, query, "absent"), [])
