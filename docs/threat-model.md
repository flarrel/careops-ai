# Threat model

## Assets

- source provenance and document versions;
- synthetic claim fixtures;
- model prompts, outputs, and trace records; and
- tool permissions and API contracts.

## Main threats and controls

| Threat | Control |
|---|---|
| Prompt injection in retrieved text | Treat retrieved text as untrusted; never allow it to alter tool permissions. |
| Unsupported answer | Require resolvable source IDs and abstain when evidence is missing. |
| Unauthorized tool use | Allow-list typed read-only tools and enforce a step budget. |
| Sensitive data in traces | Store redacted summaries and hashes; reject real records. |
| Malformed or oversized input | Validate manifest, file type, size, schema, and checksum before processing. |
| Silent model/retrieval failure | Return typed errors, write an error trace, and test failure paths. |

The MVP has no write tools, external messaging, shell execution, arbitrary SQL, or external provider calls.
