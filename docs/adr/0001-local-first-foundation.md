# ADR 0001: Local-first foundation

## Decision

Keep the maintained CareOps path local-first with Python, FastAPI, SQLite, files, and a lightweight UI. Use public provenance-recorded sources and project-owned synthetic claims.

## Reasons

- It keeps data handling inspectable and avoids billing or account dependencies.
- It supports reproducible tests on the available Apple Silicon host.
- It leaves named vendors and external systems as honest, separately gated modules.

## Consequences

The first checkpoint will not claim cloud, vendor, hospital, client, or user evidence. Resource-heavy or unavailable tools are deferred until a later checkpoint has a concrete acceptance test.
