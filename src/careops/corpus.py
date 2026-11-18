"""Reproduce the local corpus audit and verify the frozen evaluation inputs."""

import argparse
import json
import re
import unicodedata
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

SOURCES = Path("data/manifests/policy-sources.json")
DEV_CLAIMS = Path("data/samples/claims-development.jsonl")
DEV_CASES = Path("evals/development.jsonl")
SEAL = Path("evals/sealed.sha256")
SEALED_FILES = {"claims.jsonl", "cases.jsonl"}
RAW = Path("data/raw")


def digest(data):
    return sha256(data).hexdigest()


def file_digest(path):
    return digest(path.read_bytes())


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_sources():
    rows = json.loads(SOURCES.read_text(encoding="utf-8"))
    ids = [row["source"]["source_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z0-9-]+", item) for item in ids):
        raise ValueError("Source IDs must be unique PDF basenames")
    return rows


def check_pdf(data, row):
    source = row["source"]
    if not data.startswith(b"%PDF-") or len(data) != row["bytes"]:
        raise ValueError(f"{source['source_id']}: unexpected PDF signature or byte count")
    if digest(data) != source["sha256"]:
        raise ValueError(f"{source['source_id']}: SHA-256 mismatch")


def fetch_sources(rows, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = row["source"]
        path = directory / f"{source['source_id']}.pdf"
        if path.exists():
            check_pdf(path.read_bytes(), row)
            continue
        with urlopen(source["source_url"], timeout=60) as response:
            data = response.read(row["bytes"] + 1)
        check_pdf(data, row)
        with NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def normalize_page(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def iter_pdf_pages(row, directory):
    from pypdf import PdfReader

    source_id = row["source"]["source_id"]
    path = directory / f"{source_id}.pdf"
    check_pdf(path.read_bytes(), row)
    pdf = PdfReader(path)
    if len(pdf.pages) != row["pdf_pages"]:
        raise ValueError(f"{source_id}: PDF page total changed")
    for number in range(1, row["pages_in_scope"] + 1):
        yield number, pdf.pages[number - 1].extract_text() or ""


def inspect_pdf(row, directory):
    stats = {}
    for number, raw in iter_pdf_pages(row, directory):
        text = normalize_page(raw)
        stats[number] = (len(text), digest(text.encode("utf-8")), text.count("�"))
    lengths = [length for length, _, _ in stats.values()]
    counts = Counter(fingerprint for length, fingerprint, _ in stats.values() if length)
    return {
        "source_id": row["source"]["source_id"],
        "pdf_pages": row["pdf_pages"],
        "pages_inspected": len(lengths),
        "empty_pages": [number for number, (length, _, _) in stats.items() if length == 0],
        "short_pages": [number for number, (length, _, _) in stats.items() if 0 < length < 100],
        "duplicate_page_copies": sum(count - 1 for count in counts.values()),
        "replacement_characters": sum(count for _, _, count in stats.values()),
        "characters_per_page": {"min": min(lengths), "median": median(lengths), "max": max(lengths)},
    }


def validate_cases(cases, claims, sources, split):
    claim_by_id = {row["claim_id"]: row for row in claims}
    if len(claim_by_id) != len(claims):
        raise ValueError(f"{split}: duplicate claim ID")
    for row in claims:
        if (
            row["synthetic"] is not True
            or not row["claim_id"].startswith("SYN-")
            or not row["service_code"].startswith("SYN-")
            or type(row["amount_idr"]) is not int
            or row["amount_idr"] <= 0
        ):
            raise ValueError(f"{split}: invalid synthetic claim {row['claim_id']}")
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError(f"{split}: duplicate case ID")
    if len({normalize_page(row["question"]).casefold() for row in cases}) != len(cases):
        raise ValueError(f"{split}: duplicate question")
    allowed = {row["source"]["source_id"]: row for row in sources if row["case_split"] == split}
    for row in cases:
        kind = row["kind"]
        expected = row["expected"]
        if expected["status"] not in ("answer", "review", "abstain", "escalate"):
            raise ValueError(f"{row['case_id']}: invalid outcome")
        if kind == "policy":
            if expected["status"] != "answer" or not expected["evidence"]:
                raise ValueError(f"{row['case_id']}: answerable case needs evidence")
            for citation in expected["evidence"]:
                source = allowed.get(citation["source_id"])
                if source is None or not 1 <= citation["pdf_page"] <= source["pages_in_scope"]:
                    raise ValueError(f"{row['case_id']}: citation outside split or page range")
        elif kind == "tool":
            lookup = expected["lookup"]
            if lookup["claim_id"] in claim_by_id:
                claim = claim_by_id[lookup["claim_id"]]
                if any(claim[key] != value for key, value in lookup["fields"].items()):
                    raise ValueError(f"{row['case_id']}: expected fields differ from fixture")
            elif expected["status"] != "abstain" or lookup["fields"]:
                raise ValueError(f"{row['case_id']}: invalid missing-claim expectation")
        elif kind not in ("unanswerable", "injection") or expected["status"] != "abstain":
            raise ValueError(f"{row['case_id']}: invalid case type or abstention expectation")


def audit(sources, directory):
    claims, cases = read_jsonl(DEV_CLAIMS), read_jsonl(DEV_CASES)
    validate_cases(cases, claims, sources, "development")
    pdfs = [inspect_pdf(row, directory) for row in sources]
    return {
        "corpus": {
            "documents": len(pdfs),
            "bytes": sum(row["bytes"] for row in sources),
            "scoped_pages": sum(row["pages_inspected"] for row in pdfs),
            "families": dict(Counter(row["family"] for row in sources)),
            "extraction": pdfs,
        },
        "development": {
            "synthetic_claims": len(claims),
            "amount_idr": {
                "min": min(row["amount_idr"] for row in claims),
                "median": median(row["amount_idr"] for row in claims),
                "max": max(row["amount_idr"] for row in claims),
            },
            "claim_templates": dict(Counter(row["template"] for row in claims)),
            "cases_by_kind": dict(Counter(row["kind"] for row in cases)),
            "case_outcomes": dict(Counter(row["expected"]["status"] for row in cases)),
        },
    }


def verify_seal(sources, directory):
    path = directory / "manifest.json"
    if file_digest(path) != SEAL.read_text(encoding="ascii").strip():
        raise ValueError("Sealed manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["corpus_sha256"] != file_digest(SOURCES) or manifest["files"].keys() != SEALED_FILES:
        raise ValueError("Sealed manifest has unexpected inputs")
    for name, expected in manifest["files"].items():
        if file_digest(directory / name) != expected:
            raise ValueError(f"Sealed file SHA-256 mismatch: {name}")

    claims, cases = read_jsonl(directory / "claims.jsonl"), read_jsonl(directory / "cases.jsonl")
    validate_cases(cases, claims, sources, "sealed")
    if manifest["counts"] != {"claims": len(claims), "cases": len(cases)}:
        raise ValueError("Sealed manifest counts disagree with payloads")
    families = {row["family"] for row in sources if row["case_split"] == "development"}
    if any(row["case_split"] == "sealed" and row["family"] in families for row in sources):
        raise ValueError("Document family overlaps the held-out split")

    dev_claims, dev_cases = read_jsonl(DEV_CLAIMS), read_jsonl(DEV_CASES)
    validate_cases(dev_cases, dev_claims, sources, "development")
    for dev, sealed, key in (
        (dev_claims, claims, "claim_id"),
        (dev_claims, claims, "template"),
        (dev_cases, cases, "case_id"),
    ):
        if {row[key] for row in dev} & {row[key] for row in sealed}:
            raise ValueError(f"Development and sealed splits share {key}")
    dev_questions = {normalize_page(row["question"]).casefold() for row in dev_cases}
    if any(normalize_page(row["question"]).casefold() in dev_questions for row in cases):
        raise ValueError("Development and sealed splits share a question")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "audit"):
        commands.add_parser(name).add_argument("--raw", type=Path, default=RAW)
    commands.add_parser("verify-seal").add_argument("directory", type=Path)
    args = parser.parse_args()
    sources = load_sources()
    if args.command == "fetch":
        fetch_sources(sources, args.raw)
    elif args.command == "audit":
        print(json.dumps(audit(sources, args.raw), indent=2))
    else:
        verify_seal(sources, args.directory)
        print("Seal and split verified")


if __name__ == "__main__":
    main()
