"""Build balanced routing examples from grouped synthetic claim fixtures."""

from dataclasses import dataclass
from hashlib import sha256

from careops.agent_tools import claim_issues
from careops.model_lab.fixtures import split_claims

PROMPT_VERSION = "claim-routing-v1"
LABELS = ("review clear", "escalate invoice", "escalate code", "escalate amount")
PROMPT = """Route this synthetic claim for human review. Output exactly one label.
Use escalate invoice when the invoice is absent.
Use escalate code when the record code and the visit note code differ.
Use escalate amount when the amount exceeds the simulation limit.
Use review clear when all three checks pass.
"""
RULE_RESULTS = {
    (): "review clear",
    ("Missing fixture attachment: invoice",): "escalate invoice",
    ("Visit note and fixture service codes differ",): "escalate code",
    ("Fixture amount exceeds the supplied simulation limit",): "escalate amount",
}


@dataclass(frozen=True)
class Example:
    sample_id: str
    claim_id: str
    prompt: str
    alternate: str
    label: str


def examples(rows):
    splits = {}
    if any(row.get("synthetic") is not True for row in rows):
        raise ValueError("The benchmark accepts only synthetic development records")
    for split, claims in split_claims(rows).items():
        split_cases = []
        for claim in claims:
            for variant in range(3):
                split_cases.extend(variants(claim, variant))
        splits[split] = split_cases
    return splits


def variants(claim, variant):
    claim_id = claim["claim_id"]
    seed = int.from_bytes(sha256(f"{claim_id}:{variant}".encode()).digest()[:4])
    amount = claim["amount_idr"] + variant * 17240
    gap = 3000 + seed % 53000
    code = f"X{100 + seed % 899}"
    for idx, condition in enumerate(LABELS):
        invoice = condition != "escalate invoice"
        note = f"X{100 + (seed + 37) % 899}" if condition == "escalate code" else code
        limit = amount - gap if condition == "escalate amount" else amount + gap
        attachments = [f"visit-note-with-code-{note}"]
        if invoice:
            attachments.append("invoice")
        record = {
            **claim, "amount_idr": amount, "service_code": f"SYN-{code}",
            "attachments": attachments,
        }
        label = RULE_RESULTS[tuple(claim_issues(record, limit))]
        if label != condition:
            raise ValueError("Generated example does not match the fixture checklist")

        invoice_text = "present" if invoice else "absent"
        facts = (
            f"Invoice: {invoice_text}. Record code: {code}. Note code: {note}. "
            f"Amount: {amount} IDR. Simulation limit: {limit} IDR."
        )
        alternate = (
            f"The simulation limit is {limit} IDR for a claim of {amount} IDR. "
            f"The visit note shows {note}; the record shows {code}. "
            f"The invoice is {invoice_text}."
        )
        yield Example(f"{claim_id}:{variant}:{idx}", claim_id, PROMPT + facts,
                      PROMPT + alternate, label)
