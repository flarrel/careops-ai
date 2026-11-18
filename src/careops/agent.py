"""LangGraph workflow for approved synthetic claim review and policy lookup."""

from functools import cache
from threading import Lock
from typing import TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import tracing_context

from careops.agent_tools import TASK, claim_issues, load_claims, make_tools
from careops.agent_trace import JsonlTrace, fingerprint, record, start_event
from careops.contracts import validate_agent_result
from careops.corpus import DEV_CLAIMS

MAX_TOOL_CALLS = 3
MAX_ATTEMPTS_PER_TOOL = 2


class ReviewState(TypedDict, total=False):
    task: dict
    next_node: str
    pending: str
    args: dict
    attempt: int
    tool_calls: list[str]
    tool_output: dict | None
    claim: dict | None
    result: dict


def finish(status, summary, reason, *, calls, citations=None):
    return {
        "result": {"status": status, "summary": summary, "reason": reason,
                   "citations": citations or [],
                   "tool_calls": calls},
        "next_node": END,
    }


def build_graph(tools, trace, run_id, task_hash, budget):
    def emit(stage, outcome, start, **details):
        record(trace, run_id, task_hash, stage, outcome, start, **details)

    def schedule(name, args):
        return {"pending": name, "args": args, "attempt": 0, "next_node": "tool"}

    def plan(state):
        task = state["task"]
        if task["kind"] == "claim":
            return {"next_node": "approval"}
        return schedule("lookup_policy", {"query": task["query"]})

    def approval(state):
        decision = interrupt({"action": "read_synthetic_claim"})
        approved = isinstance(decision, dict) and decision.get("approved") is True
        start = start_event()
        emit("approval", "ok" if approved else "abstain", start)
        if approved:
            return schedule("lookup_synthetic_claim", {"claim_id": state["task"]["claim_id"]})
        return finish("abstain", "Synthetic lookup declined.", "approval_denied", calls=[])

    def use_tool(state):
        name = state["pending"]
        calls = state["tool_calls"]
        start = start_event()
        if name not in tools or len(calls) >= budget:
            reason = "tool_budget_exhausted" if name in tools else "tool_not_allowed"
            emit("agent", "escalate", start, error_code=reason)
            return finish("escalate", "A human reviewer must continue.", reason, calls=calls)

        attempt = state["attempt"] + 1
        calls = [*calls, name]
        try:
            output = tools[name].invoke(state["args"])
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            transient = isinstance(exc, OSError)
            reason = "tool_timeout" if isinstance(exc, TimeoutError) else "tool_failure"
            emit("tool", "error", start, tool_name=name, attempt=attempt, error_code=reason)
            if transient and attempt < MAX_ATTEMPTS_PER_TOOL and len(calls) < budget:
                return {"attempt": attempt, "tool_calls": calls, "next_node": "tool"}
            return finish("escalate", "The lookup needs human follow-up.", reason, calls=calls)

        emit("tool", "ok", start, tool_name=name, attempt=attempt)
        return {"tool_calls": calls, "tool_output": output,
                "next_node": "check_claim" if name == "lookup_synthetic_claim" else "check_policy"}

    def check_claim(state):
        claim = state["tool_output"]
        calls = state["tool_calls"]
        if claim is None:
            return finish("abstain", "Synthetic claim was not found.", "claim_not_found",
                          calls=calls)

        issues = claim_issues(claim, state["task"].get("max_amount_idr"))
        if issues:
            return {"claim": claim, **finish("escalate", ". ".join(issues) + ".",
                    "fixture_checks_failed", calls=calls)}
        policy_query = state["task"].get("policy_query")
        if policy_query:
            return {"claim": claim, **schedule("lookup_policy", {"query": policy_query})}
        return {"claim": claim, **finish("review", "Synthetic checklist ready for human review.",
                "human_review_required", calls=calls)}

    def check_policy(state):
        reply = state["tool_output"]
        calls = state["tool_calls"]
        combined = state["task"]["kind"] == "claim"
        if reply["status"] == "abstain":
            return finish("escalate" if combined else "abstain", "Policy evidence is insufficient.",
                          "policy_evidence_missing", calls=calls)

        summary = "Policy excerpt for human review. " + reply["excerpt"]
        return finish("review", summary, "human_review_required",
                      calls=calls, citations=reply["citations"])

    workflow = StateGraph(ReviewState)
    nodes = {"plan": plan, "approval": approval, "tool": use_tool,
             "check_claim": check_claim, "check_policy": check_policy}
    for name, function in nodes.items():
        workflow.add_node(name, function)
    workflow.add_edge(START, "plan")
    destinations = {END: END, **{name: name for name in nodes if name != "plan"}}
    for name in nodes:
        workflow.add_conditional_edges(name, lambda state: state["next_node"], destinations)
    return workflow.compile(checkpointer=MemorySaver())


class ReviewRun:
    def __init__(self, task, tools, trace, budget):
        self.run_id = uuid4().hex
        self.task_hash = fingerprint(task)
        self.trace = trace
        self.lock = Lock()
        self.graph = build_graph(tools, trace, self.run_id, self.task_hash, budget)
        self.config = {"configurable": {"thread_id": self.run_id}, "recursion_limit": 16}
        self.response = self._advance({"task": task, "tool_calls": []})

    def _advance(self, graph_input):
        start = start_event()
        with tracing_context(enabled=False):
            state = self.graph.invoke(graph_input, config=self.config)
        if "result" not in state:
            record(self.trace, self.run_id, self.task_hash, "approval", "pending", start)
            return {"run_id": self.run_id, "status": "awaiting_approval"}

        result = state["result"]
        validate_agent_result(result)
        outcome = "ok" if result["status"] == "review" else result["status"]
        record(self.trace, self.run_id, self.task_hash, "agent", outcome, start,
               citation_ids=result["citations"])
        return {"run_id": self.run_id, "status": "complete", "result": result,
                "lookup": state.get("claim")}

    def resume(self, approve: bool):
        with self.lock:
            if self.response["status"] != "awaiting_approval":
                raise ValueError("This run is not awaiting approval")
            if type(approve) is not bool:
                raise ValueError("Approval must be a boolean")
            try:
                self.response = self._advance(Command(resume={"approved": approve}))
            except OSError:
                self.response = {"run_id": self.run_id, "status": "failed"}
                raise
            return self.response


class ReviewAgent:
    def __init__(self, read_claim, answer_policy, trace=None, *, budget=MAX_TOOL_CALLS):
        if type(budget) is not int or not 1 <= budget <= MAX_TOOL_CALLS:
            raise ValueError("The tool budget must be between one and three attempts")
        self.tools = make_tools(read_claim, answer_policy)
        self.trace = trace if trace is not None else JsonlTrace()
        self.budget = budget

    def start(self, task):
        request = TASK.validate_python(task)
        return ReviewRun(request.model_dump(), self.tools, self.trace, self.budget)


def load_agent(answer_policy, trace=None, *, claims_path=DEV_CLAIMS):
    @cache
    def claims():
        return load_claims(claims_path)

    return ReviewAgent(lambda claim_id: claims().get(claim_id), answer_policy, trace)
