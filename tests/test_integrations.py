import importlib.util
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

INTEGRATIONS = all(importlib.util.find_spec(name) for name in ("openpyxl", "pydantic"))


@unittest.skipUnless(INTEGRATIONS, "Spreadsheet tests require the integrations and rag extras")
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        from careops.agent_tools import load_claims

        directory = self.enterContext(TemporaryDirectory())
        self.workbook = Path(directory) / "claims.xlsx"
        self.claims = load_claims()

    def test_xlsx_round_trip_and_normalized_agent_input(self):
        from careops.agent_tools import load_claims
        from careops.integrations.spreadsheet import read_xlsx, write_jsonl, write_xlsx

        write_xlsx(self.claims.values(), self.workbook)
        claims = read_xlsx(self.workbook)
        self.assertEqual(claims, list(self.claims.values()))
        output = self.workbook.with_suffix(".jsonl")
        write_jsonl(claims, output)
        self.assertEqual(load_claims(output), self.claims)

    def test_xlsx_rejects_formula_duplicate_and_non_synthetic_rows(self):
        from openpyxl import load_workbook
        from careops.integrations.spreadsheet import read_xlsx, write_xlsx

        for row, column, value in ((2, 5, "=7+1"), (3, 1, "SYN-D-001"), (2, 2, False),
                                   (2, 7, "unexpected")):
            with self.subTest(row=row, column=column, value=value):
                write_xlsx(self.claims.values(), self.workbook)
                workbook = load_workbook(self.workbook)
                workbook.active.cell(row, column, value)
                workbook.save(self.workbook)
                workbook.close()
                with self.assertRaises(ValueError):
                    read_xlsx(self.workbook)

    def test_contracts_preserve_amount_and_prevent_unmarked_intake(self):
        from pydantic import ValidationError
        from careops.integrations.healthcare import IntakeRecord, dataverse_task, fhir_claim

        source = self.claims["SYN-D-001"]
        record = {
            "schema_version": "careops-intake-1", "source": "emr", "synthetic": True,
            "reference": source.claim_id, "service": source.service_type,
            "local_service_code": source.service_code, "amount": source.amount_idr,
            "currency": "IDR", "attachment_codes": source.attachments,
        }
        for channel in ("his", "emr"):
            self.assertEqual(IntakeRecord.model_validate({**record, "source": channel}).normalize(),
                             source)
        with self.assertRaises(ValidationError):
            IntakeRecord.model_validate({**record, "currency": "USD"})
        with self.assertRaises(ValidationError):
            IntakeRecord.model_validate({**record, "patient": "example"})
        with self.assertRaises(ValueError):
            IntakeRecord.model_validate({**record, "reference": "SYN-F-001"}).normalize()

        exported = fhir_claim(source, date(2026, 1, 1))
        self.assertEqual(exported["status"], "draft")
        self.assertEqual(exported["patient"]["reference"], "Patient/synthetic-demo")
        self.assertEqual(exported["item"][0]["net"], exported["total"])
        self.assertEqual(exported["total"], {"value": source.amount_idr, "currency": "IDR"})
        task = dataverse_task(source)
        self.assertEqual(task["path"], "/api/data/v9.2/tasks")
        self.assertIn("invoice", task["body"]["description"])
        self.assertIsNone(dataverse_task(source.model_copy(update={"attachments": [
            "invoice", "visit-note",
        ]})))
