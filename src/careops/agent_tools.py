"""Typed inputs and read-only lookups for the policy and synthetic claim tools."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from careops.answer import MAX_QUERY_CHARS
from careops.corpus import DEV_CLAIMS, read_jsonl

CLAIM_ID = r"^SYN-[A-Z]-[0-9]{3}$"
Question = Annotated[str, Field(min_length=1, max_length=MAX_QUERY_CHARS)]
ClaimId = Annotated[str, Field(pattern=CLAIM_ID)]
Amount = Annotated[int, Field(gt=0, le=10**12)]


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)


class ClaimLookup(StrictInput):
    claim_id: ClaimId


class PolicyLookup(StrictInput):
    query: Question


class ClaimTask(ClaimLookup):
    kind: Literal["claim"]
    max_amount_idr: Amount | None = None
    policy_query: Question | None = None


class PolicyTask(PolicyLookup):
    kind: Literal["policy"]


TASK = TypeAdapter(Annotated[ClaimTask | PolicyTask, Field(discriminator="kind")])


class SyntheticClaim(ClaimLookup):
    synthetic: Literal[True]
    service_type: Literal["outpatient", "day-care", "inpatient", "diagnostic", "specialist"]
    service_code: str = Field(pattern=r"^SYN-[A-Z0-9-]+$")
    amount_idr: Amount
    attachments: list[Annotated[str, Field(pattern=r"^[a-zA-Z0-9-]{1,64}$")]]


class PolicyFinding(StrictInput):
    status: Literal["answer", "abstain"]
    excerpt: str = Field(min_length=1)
    citations: list[Annotated[str, Field(
        pattern=r"^[a-z0-9-]+:p[1-9][0-9]*:[0-9]+-[0-9]+:[a-f0-9]{12}$"
    )]]


def load_claims(path=DEV_CLAIMS):
    records = {}
    for row in read_jsonl(path):
        if row.get("synthetic") is not True:
            raise ValueError("The claim store accepts only labeled synthetic fixtures")
        # Template names are evaluation metadata and never enter the workflow.
        facts = SyntheticClaim.model_validate({key: value for key, value in row.items()
                                               if key != "template"})
        if facts.claim_id in records:
            raise ValueError("Duplicate synthetic claim ID")
        records[facts.claim_id] = facts
    return records


def make_tools(read_claim, answer_policy):
    from langchain_core.tools import StructuredTool

    def lookup_synthetic_claim(claim_id):
        facts = read_claim(claim_id)
        return SyntheticClaim.model_validate(facts).model_dump() if facts is not None else None

    def lookup_policy(query):
        reply = answer_policy(query)
        result = PolicyFinding(
            status=reply["status"], excerpt=reply["answer"],
            citations=[citation["chunk_id"] for citation in reply["citations"]],
        )
        if (result.status == "answer") != bool(result.citations):
            raise ValueError("Policy status and citations disagree")
        return result.model_dump()

    return {
        "lookup_synthetic_claim": StructuredTool.from_function(
            lookup_synthetic_claim, args_schema=ClaimLookup,
            description="Read a single project-owned synthetic claim fixture.",
        ),
        "lookup_policy": StructuredTool.from_function(
            lookup_policy, args_schema=PolicyLookup,
            description="Read a cited policy passage from the local source corpus.",
        ),
    }


def claim_issues(claim, limit):
    attachments = claim["attachments"]
    required = {"invoice"}
    required.update({
        "outpatient": {"visit-note"},
        "day-care": {"visit-note"},
        "inpatient": {"signed-discharge-summary"},
        "specialist": {"specialist-note", "referral"},
        "diagnostic": {"diagnostic-note"},
    }[claim["service_type"]])
    note_codes = [item.removeprefix("visit-note-with-code-") for item in attachments
                  if item.startswith("visit-note-with-code-")]
    observed = set(attachments)
    if note_codes:
        observed.add("visit-note")
    issues = [f"Missing fixture attachment: {item}" for item in sorted(required - observed)]
    if any(f"SYN-{code}" != claim["service_code"] for code in note_codes):
        issues.append("Visit note and fixture service codes differ")
    if len(set(attachments)) != len(attachments):
        issues.append("Duplicate attachment in the fixture")
    if limit is not None and claim["amount_idr"] > limit:
        issues.append("Fixture amount exceeds the supplied simulation limit")
    return issues
