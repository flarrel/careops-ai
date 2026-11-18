import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from careops import corpus


class CorpusTests(unittest.TestCase):
    def test_development_references_match_source_scope_and_claims(self):
        corpus.validate_cases(
            corpus.read_jsonl(corpus.DEV_CASES),
            corpus.read_jsonl(corpus.DEV_CLAIMS),
            corpus.load_sources(),
            "development",
        )

    def test_pdf_integrity_rejects_different_bytes_of_the_same_size(self):
        pdf = b"%PDF-example"
        row = {"source": {"source_id": "example", "sha256": corpus.digest(pdf)}, "bytes": len(pdf)}
        corpus.check_pdf(pdf, row)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            corpus.check_pdf(b"%PDF-eXample", row)

    def test_seal_rejects_tampering_or_template_overlap(self):
        sources = corpus.load_sources()
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            seal = directory / "public.sha256"
            cases = directory / "cases.jsonl"
            cases.write_text("")
            claims = directory / "claims.jsonl"
            claim = dict(corpus.read_jsonl(corpus.DEV_CLAIMS)[0], claim_id="SYN-H-990")
            manifest = directory / "manifest.json"

            def pin():
                manifest.write_text(json.dumps({
                    "corpus_sha256": corpus.file_digest(corpus.SOURCES),
                    "files": {p.name: corpus.file_digest(p) for p in (cases, claims)},
                    "counts": {"claims": 1, "cases": 0},
                }))
                seal.write_text(corpus.file_digest(manifest))

            with patch.object(corpus, "SEAL", seal):
                claims.write_text(json.dumps(claim) + "\n")
                pin()
                with self.assertRaisesRegex(ValueError, "share template"):
                    corpus.verify_seal(sources, directory)

                claim["template"] = "test-only"
                claims.write_text(json.dumps(claim) + "\n")
                pin()
                corpus.verify_seal(sources, directory)
                claims.write_text("")
                with self.assertRaisesRegex(ValueError, "Sealed file SHA-256 mismatch"):
                    corpus.verify_seal(sources, directory)
