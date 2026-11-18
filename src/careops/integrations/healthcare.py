"""Map project-owned synthetic fixtures to bounded interoperability contracts."""

from datetime import date
from typing import Literal

from pydantic import Field

from careops.agent_tools import Amount, ClaimId, StrictInput, SyntheticClaim, claim_issues

DEMO = "https://example.org/careops/synthetic"


class IntakeRecord(StrictInput):
    schema_version: Literal["careops-intake-1"]
    source: Literal["his", "emr"]
    synthetic: Literal[True]
    reference: ClaimId
    service: Literal["outpatient", "day-care", "inpatient", "diagnostic", "specialist"]
    local_service_code: str
    amount: Amount
    currency: Literal["IDR"]
    attachment_codes: list[str] = Field(max_length=32)

    def normalize(self):
        if not self.reference.startswith("SYN-D-") or self.synthetic is not True:
            raise ValueError("Expected a synthetic development record")
        return SyntheticClaim(
            claim_id=self.reference, synthetic=True, service_type=self.service,
            service_code=self.local_service_code, amount_idr=self.amount,
            attachments=self.attachment_codes,
        )


def fhir_claim(claim: SyntheticClaim, created: date):
    money = {"value": claim.amount_idr, "currency": "IDR"}
    return {
        "resourceType": "Claim",
        "id": claim.claim_id,
        "meta": {"tag": [{"system": DEMO, "code": "synthetic"}]},
        "identifier": [{"system": DEMO, "value": claim.claim_id}],
        "status": "draft",
        "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/claim-type",
                             "code": "institutional"}]},
        "use": "claim",
        "patient": {"reference": "Patient/synthetic-demo"},
        "created": created.isoformat(),
        "provider": {"reference": "Organization/synthetic-provider"},
        "priority": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/processpriority",
                                 "code": "normal"}]},
        "insurance": [{"sequence": 1, "focal": True, "coverage": {
            "reference": "Coverage/synthetic-demo",
        }}],
        "item": [{"sequence": 1, "productOrService": {
            "coding": [{"system": DEMO + "/services", "code": claim.service_code}],
        }, "net": money}],
        "total": money,
    }


def dataverse_task(claim: SyntheticClaim):
    issues = claim_issues(claim.model_dump(), None)
    if not issues:
        return None
    return {
        "method": "POST",
        "path": "/api/data/v9.2/tasks",
        "headers": {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
        },
        "body": {
            "subject": "Synthetic claim review " + claim.claim_id,
            "description": "\n".join(issues),
        },
    }
