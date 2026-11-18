"""Validate a synthetic preview against the upstream HL7 FHIR R4 JSON schema."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from jsonschema import Draft6Validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schema", type=Path)
    parser.add_argument("preview", type=Path)
    args = parser.parse_args()
    with ZipFile(args.schema) as archive:
        schema = json.loads(archive.read("fhir.schema.json"))
    preview = json.loads(args.preview.read_bytes())
    # Validate only the Claim resource while keeping its local references resolvable.
    schema.pop("oneOf", None)
    schema["allOf"] = [{"$ref": "#/definitions/Claim"}]
    Draft6Validator(schema).validate(preview["fhir_claim"])
    print("Synthetic Claim passed the HL7 FHIR R4 JSON schema")


if __name__ == "__main__":
    main()
