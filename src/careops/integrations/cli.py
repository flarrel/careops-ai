"""Prepare synthetic claim fixtures and offline interoperability payloads."""

import argparse
import json
from datetime import date
from pathlib import Path
from zipfile import BadZipFile

from openpyxl.utils.exceptions import InvalidFileException

from careops.agent_tools import load_claims
from careops.corpus import DEV_CLAIMS
from careops.integrations.healthcare import IntakeRecord, dataverse_task, fhir_claim
from careops.integrations.spreadsheet import read_xlsx, write_jsonl, write_xlsx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-xlsx")
    export.add_argument("workbook", type=Path)
    export.add_argument("--claims", type=Path, default=DEV_CLAIMS)
    intake = commands.add_parser("import-xlsx")
    intake.add_argument("workbook", type=Path)
    intake.add_argument("output", type=Path)
    normalize = commands.add_parser("normalize-intake")
    normalize.add_argument("input", type=Path)
    normalize.add_argument("output", type=Path)
    contracts = commands.add_parser("preview")
    contracts.add_argument("claim_id")
    contracts.add_argument("--claims", type=Path, default=DEV_CLAIMS)
    contracts.add_argument("--date", type=date.fromisoformat, default=date(2026, 1, 1))
    args = parser.parse_args()
    try:
        if args.command == "export-xlsx":
            write_xlsx(load_claims(args.claims).values(), args.workbook)
        elif args.command == "import-xlsx":
            claims = read_xlsx(args.workbook)
            write_jsonl(claims, args.output)
            print(f"Imported {len(claims)} synthetic claims")
        elif args.command == "normalize-intake":
            if args.input.stat().st_size > 16 * 1024:
                raise ValueError("Intake exceeds 16 KB")
            claim = IntakeRecord.model_validate_json(args.input.read_bytes()).normalize()
            write_jsonl([claim], args.output)
        else:
            claim = load_claims(args.claims).get(args.claim_id)
            if claim is None:
                raise ValueError("Synthetic claim ID was not found")
            print(json.dumps({"fhir_claim": fhir_claim(claim, args.date),
                              "dataverse_task": dataverse_task(claim)}, indent=2))
    except (ValueError, OSError, BadZipFile, InvalidFileException) as exc:
        # Schema and file-read errors can contain user-supplied values or local paths.
        if type(exc) is ValueError:
            parser.error(str(exc))
        parser.error("Could not read or write the synthetic fixture data")


if __name__ == "__main__":
    main()
