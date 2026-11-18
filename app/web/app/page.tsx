"use client";

import { type FormEvent, useCallback, useEffect, useState } from "react";
import { api, type PendingReview, type PolicyAnswer, type Review, type Run } from "./contracts";

const outcomes: Record<string, string> = {
  answer: "Cited answer", abstain: "Abstained", review: "Ready for human review",
  escalate: "Needs human follow-up", error: "Service error",
};

const formatAmount = new Intl.NumberFormat("en", { style: "currency", currency: "IDR",
  maximumFractionDigits: 0 });

function ErrorMessage({ message }: { message: string }) {
  return message && <p className="notice error" role="alert">{message}</p>;
}

export default function Home() {
  const [mode, setMode] = useState<"policy" | "claim">("policy");
  const [answer, setAnswer] = useState<PolicyAnswer>();
  const [pending, setPending] = useState<PendingReview>();
  const [review, setReview] = useState<Review>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [runs, setRuns] = useState<Run[]>();
  const [historyError, setHistoryError] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    setHistoryError("");
    try {
      setRuns(await api<Run[]>("runs?limit=10"));
    } catch (cause) {
      setHistoryError((cause as Error).message);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  async function perform(action: () => Promise<void>, recordsRun: boolean) {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setBusy(false);
      if (recordsRun) void refresh();
    }
  }

  function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = new FormData(event.currentTarget).get("query");
    setAnswer(undefined);
    void perform(async () => setAnswer(await api("answer", { query })), true);
  }

  function startReview(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const limit = data.get("limit");
    setReview(undefined);
    void perform(async () => setPending(await api("reviews", {
      claim_id: data.get("claim"),
      max_amount_idr: limit ? Number(limit) : null,
      policy_query: String(data.get("policy") ?? "").trim() || null,
    })), false);
  }

  function decide(approved: boolean) {
    if (!pending) return;
    void perform(async () => {
      setReview(await api(`reviews/${pending.run_id}/decision`, { approved }));
      setPending(undefined);
    }, true);
  }

  return (
    <>
      <header className="masthead"><div className="frame brand">
        <span className="brand-mark" aria-hidden="true">+</span>
        <span>CareOps AI</span>
        <span className="brand-context">Local workspace</span>
      </div></header>
      <main className="frame">
        <div className="intro">
          <h1>Policy and claim review</h1>
          <p>Find cited policy excerpts and run read-only checks on synthetic claims.</p>
        </div>
        <div className="panels">
          <section className="panel workflow" aria-label="Review workflow">
            <div className="switch" role="group" aria-label="Workflow">
              <button aria-pressed={mode === "policy"} disabled={busy}
                onClick={() => { setError(""); setMode("policy"); }}>Policy lookup</button>
              <button aria-pressed={mode === "claim"} disabled={busy}
                onClick={() => { setError(""); setMode("claim"); }}>Synthetic review</button>
            </div>
            <ErrorMessage message={error} />
            {mode === "policy" ? <>
              <h2>Find policy evidence</h2>
              <form onSubmit={ask}>
                <label htmlFor="question">Policy question</label>
                <textarea id="question" name="query" rows={4} maxLength={500} required
                  placeholder="Tanyakan kebijakan atau prosedur…" />
                <button className="primary" disabled={busy} type="submit">
                  {busy ? "Looking up evidence…" : "Find evidence"}
                </button>
              </form>
              {answer && <div className="result" role="status">
                <span className={`status ${answer.status}`}>{outcomes[answer.status]}</span>
                <p>{answer.answer}</p>
                {answer.citations.length > 0 && <ul className="sources">
                  {answer.citations.map(citation => <li key={citation.chunk_id}>
                    <a target="_blank" rel="noreferrer" href={citation.url}>{citation.title}</a>
                    <small>{citation.location}</small>
                  </li>)}
                </ul>}
              </div>}
            </> : <>
              <h2>Review a synthetic claim</h2>
              {pending ? <div className="notice">
                <p>Allow read access to this synthetic fixture for human review?</p>
                <div className="actions">
                  <button className="primary" disabled={busy} onClick={() => decide(true)}>Allow lookup</button>
                  <button disabled={busy} onClick={() => decide(false)}>Decline lookup</button>
                  {error && <button disabled={busy} onClick={() => { setPending(undefined); setError(""); }}>
                    Start over
                  </button>}
                </div>
                {busy && <small role="status">Recording decision…</small>}
              </div> : <form onSubmit={startReview}>
                <label htmlFor="claim">Synthetic claim ID</label>
                <input id="claim" name="claim" placeholder="SYN-D-001" maxLength={9} required />
                <label htmlFor="limit">Simulation limit (IDR) <small>Optional</small></label>
                <input id="limit" name="limit" type="number" min={1} max={1e12} step={1} />
                <label htmlFor="policy">Policy question <small>Optional</small></label>
                <input id="policy" name="policy" maxLength={500} />
                <button className="primary" disabled={busy} type="submit">
                  {busy ? "Starting review…" : "Request review"}
                </button>
              </form>}
              {review && <div className="result" role="status">
                <span className={`status ${review.result.status}`}>{outcomes[review.result.status]}</span>
                <p>{review.result.summary}</p>
                {review.lookup && <dl>
                  <dt>Claim</dt><dd>{review.lookup.claim_id}</dd>
                  <dt>Amount</dt><dd>{formatAmount.format(review.lookup.amount_idr)}</dd>
                  <dt>Service code</dt><dd>{review.lookup.service_code}</dd>
                  <dt>Attachments</dt><dd>{review.lookup.attachments.join(", ") || "None"}</dd>
                </dl>}
                {review.result.citations.length > 0 && <ul className="sources">
                  {review.result.citations.map(citation => <li key={citation}><code>{citation}</code></li>)}
                </ul>}
                {review.result.tool_calls.length > 0 &&
                  <small>Tools · {review.result.tool_calls.join(", ")}</small>}
              </div>}
            </>}
          </section>
          <section className="panel history" aria-labelledby="history">
            <div className="panel-heading">
              <h2 id="history">Run history</h2>
              <button disabled={refreshing} onClick={() => void refresh()}>Refresh</button>
            </div>
            <ErrorMessage message={historyError} />
            {runs ? runs.length ? <table>
              <thead><tr><th scope="col">Workflow</th><th scope="col">Outcome</th><th scope="col">Duration</th></tr></thead>
              <tbody>{runs.map(run => <tr key={run.run_id}>
                <td>{run.kind === "policy" ? "Policy" : "Claim"}
                  <small><time dateTime={run.started_at}>
                    {new Date(run.started_at).toLocaleString("en", { dateStyle: "short", timeStyle: "short" })}
                  </time></small>
                </td>
                <td><span className={`status ${run.status}`}>{outcomes[run.status] ?? run.status}</span></td>
                <td>{Math.round(run.duration_ms)} ms</td>
              </tr>)}</tbody>
            </table> : <p className="muted">No completed runs yet.</p> :
              refreshing && <p className="muted" role="status">Loading run history…</p>}
          </section>
        </div>
      </main>
    </>
  );
}
