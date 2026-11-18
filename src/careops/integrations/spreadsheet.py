"""Export and validate a bounded XLSX sheet of structured synthetic claims."""

from contextlib import closing
from tempfile import NamedTemporaryFile
from pathlib import Path
from zipfile import ZipFile

from openpyxl import Workbook, load_workbook
from pydantic import ValidationError

from careops.agent_tools import SyntheticClaim

COLUMNS = ("claim_id", "synthetic", "service_type", "service_code", "amount_idr", "attachments")
SHEET = "Claims"
MAX_ROWS = 500
MAX_WORKBOOK_BYTES = 2 * 1024 * 1024
MAX_EXPANDED_BYTES = 8 * 1024 * 1024


def write_xlsx(claims, target):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET
    sheet.freeze_panes = "A2"
    sheet.append(COLUMNS)
    for claim in claims:
        values = claim.model_dump()
        values["attachments"] = "|".join(claim.attachments)
        sheet.append([values[key] for key in COLUMNS])
    workbook.save(target)


def read_xlsx(path):
    if path.suffix.lower() != ".xlsx" or path.stat().st_size > MAX_WORKBOOK_BYTES:
        raise ValueError("Expected an XLSX file of at most 2 MB")
    with ZipFile(path) as archive:
        if sum(member.file_size for member in archive.infolist()) > MAX_EXPANDED_BYTES:
            raise ValueError("Uncompressed workbook exceeds 8 MB")
    with closing(load_workbook(path, read_only=True, keep_links=False)) as workbook:
        if workbook.sheetnames != [SHEET]:
            raise ValueError("Expected one worksheet named Claims")
        sheet = workbook[SHEET]
        if sheet.max_row is not None and sheet.max_row > MAX_ROWS + 1:
            raise ValueError("Worksheet exceeds 500 data rows")
        # Read actual worksheet cells, including extra columns hidden by stale dimension metadata.
        sheet.reset_dimensions()
        rows = sheet.iter_rows()
        if tuple(cell.value for cell in next(rows, ())) != COLUMNS:
            raise ValueError("Worksheet headers do not match the synthetic claim template")
        seen = set()
        claims = []
        for number, cells in enumerate(rows, start=2):
            if number > MAX_ROWS + 1:
                raise ValueError("Worksheet exceeds 500 data rows")
            values = [cell.value for cell in cells]
            if not any(value is not None for value in values):
                continue
            if len(cells) > len(COLUMNS) or any(cell.data_type == "f" for cell in cells):
                raise ValueError(f"Row {number} has unexpected columns or formulas")
            row = dict(zip(COLUMNS, values))
            attachments = row.get("attachments")
            if attachments is not None and not isinstance(attachments, str):
                raise ValueError(f"Row {number} requires text attachment codes")
            row["attachments"] = attachments.split("|") if attachments else []
            if row.get("synthetic") is not True:
                raise ValueError(f"Row {number} requires the synthetic flag")
            try:
                claim = SyntheticClaim.model_validate(row)
            except ValidationError:
                raise ValueError(f"Row {number} has invalid synthetic claim fields") from None
            if not claim.claim_id.startswith("SYN-D-") or claim.claim_id in seen:
                raise ValueError(f"Row {number} has a duplicate or non-development claim ID")
            seen.add(claim.claim_id)
            claims.append(claim)
    if not claims:
        raise ValueError("Worksheet has no synthetic claims")
    return claims


def write_jsonl(claims, target):
    if target.suffix.lower() != ".jsonl":
        raise ValueError("Expected a .jsonl output path")
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            for claim in claims:
                stream.write(claim.model_dump_json() + "\n")
            stream.close()
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
