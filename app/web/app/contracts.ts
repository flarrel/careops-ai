export type PolicyAnswer = {
  status: "answer" | "abstain";
  answer: string;
  citations: { chunk_id: string; title: string; location: string; url: string }[];
};

export type PendingReview = { run_id: string; status: "awaiting_approval" };

export type Review = {
  run_id: string;
  status: "complete";
  result: {
    status: "review" | "abstain" | "escalate";
    summary: string;
    citations: string[];
    tool_calls: string[];
  };
  lookup: {
    claim_id: string;
    amount_idr: number;
    service_code: string;
    attachments: string[];
  } | null;
};

export type Run = {
  run_id: string;
  kind: "policy" | "claim";
  started_at: string;
  status: string;
  duration_ms: number;
  citation_count: number;
  tool_call_count: number;
};

export async function api<T>(path: string, body?: object): Promise<T> {
  let response: Response;
  try {
    response = await fetch("/api/" + path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      cache: "no-store",
    });
  } catch {
    throw new Error("Connection lost. Try again when the local server is reachable.");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok || data === null) {
    throw new Error(typeof data?.detail === "string" ? data.detail : "The request failed. Try again.");
  }
  return data as T;
}
