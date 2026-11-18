# Local application

FastAPI serves the policy answerer and bounded claim-review agent. The Streamlit admin and the Next.js client both call the same REST API. The browser talks to a Next.js route that forwards only the application endpoints to FastAPI.

## Run

Use Python 3.12 from the repository root. The first two commands that fetch data and model files are only needed when their verified local copies are missing.

```bash
python3 -m pip install -e '.[rag,agent,app]'
python3 -m careops.corpus fetch
python3 -m careops.answer_eval fetch-model
python3 -m uvicorn careops.api:app --host 127.0.0.1 --port 8000
```

Open the API's [OpenAPI UI](http://127.0.0.1:8000/docs), or start either client in a separate terminal.

```bash
python3 -m streamlit run app/admin.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

```bash
cd app/web
npm ci
npm run dev
```

Streamlit normally uses [port 8501](http://127.0.0.1:8501) and Next.js uses [port 3000](http://127.0.0.1:3000). The web client uses Node 22 or later. Both clients accept `CAREOPS_API_URL` as a server-side environment variable.

Ask “Siapa pengguna pedoman INA-CBG untuk metode pembayaran klaim menurut Permenkes 26/2021?” to see a cited passage. Request review for `SYN-D-001`, then allow the read-only lookup to see a missing-invoice escalation. Declining records an abstention without reading the fixture.

## API and storage

| Method and path | Behavior |
| --- | --- |
| `GET /health` | Process liveness. Policy resources load on first use. |
| `POST /answer` | Accepts `query`, returns a cited excerpt or abstention. |
| `POST /reviews` | Accepts `claim_id` and optional `max_amount_idr` and `policy_query`. Returns HTTP 202 with a pending run ID. |
| `POST /reviews/{run_id}/decision` | Accepts a JSON boolean `approved`. The decision consumes the pending run. |
| `GET /runs?limit=20` | Returns recent outcomes, timings, citation counts, and tool counts. Maximum 100. |

Pending approvals live in one API process for up to ten minutes. Serve with a single worker. Completed runs and controlled service failures are summarized in `data/runtime/runs.sqlite3`. The database stores timestamps, opaque IDs, outcome codes, and counts. Request text, claim identifiers, and results stay out of run history. The agent separately writes redacted events to `api-agent-traces.jsonl` beside the database. Set `CAREOPS_HISTORY_PATH` to use a separate database for a test run.

This is a loopback application for one operator and project-owned synthetic fixtures. Policy coverage is limited to the [measured corpus and answer path](grounded-answers.md). Claim results are checklists for human review.

## Verify

```bash
python3 -m unittest discover -s tests
```

Start FastAPI with an isolated `CAREOPS_HISTORY_PATH` and the fetched resources for the two live client checks. From the repository root, run `python3 tests/live_admin.py`. From `app/web`, run the production build before testing.

```bash
npm run build
npm run typecheck
npx playwright install chromium
npm test
```

Set `PLAYWRIGHT_CHANNEL=chrome` to use an installed Chrome instead of downloading Chromium. The browser suite starts the built Next.js server on port 3000 and calls the live FastAPI server on port 8000.
