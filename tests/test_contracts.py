import unittest

from careops.contracts import validate_agent_result, validate_evidence_packet


class ContractTests(unittest.TestCase):
    def test_valid_evidence_packet(self) -> None:
        validate_evidence_packet(
            {
                "query": "What is covered?",
                "items": [
                    {"chunk_id": "c-1", "source_id": "s-1", "text": "Coverage", "score": 0.8}
                ],
            }
        )

    def test_evidence_packet_rejects_missing_citation_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required keys"):
            validate_evidence_packet({"query": "What is covered?", "items": [{"text": "Coverage"}]})

    def test_agent_result_rejects_unknown_status(self) -> None:
        for status in ("approve", []):
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, "invalid agent status"):
                    validate_agent_result(
                        {"status": status, "summary": "", "citations": [], "tool_calls": []}
                    )


if __name__ == "__main__":
    unittest.main()
