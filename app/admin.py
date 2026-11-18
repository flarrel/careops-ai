"""Streamlit operator view for the local CareOps REST API."""

import os

import httpx
import streamlit as st

API = os.environ.get("CAREOPS_API_URL", "http://127.0.0.1:8000").rstrip("/")


def request(path, *, payload=None):
    try:
        if payload is None:
            response = httpx.get(f"{API}{path}", timeout=4, trust_env=False)
        else:
            response = httpx.post(f"{API}{path}", json=payload, timeout=60, trust_env=False)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            st.error(exc.response.json().get("detail", "The request failed."))
        except ValueError:
            st.error("The request failed.")
    except (httpx.RequestError, ValueError):
        st.error("The API is unavailable. Check the local server and try again.")
    return None


st.set_page_config(page_title="CareOps AI Admin", layout="wide")
st.title("CareOps AI Admin")
st.caption("Local policy excerpts and synthetic claim reviews.")
policy, claims, history = st.tabs(["Policy lookup", "Synthetic review", "Run history"])

with policy:
    with st.form("policy"):
        query = st.text_area("Policy question", max_chars=500)
        submit = st.form_submit_button("Find policy evidence")
    if submit:
        st.session_state.answer = request("/answer", payload={"query": query})
    result = st.session_state.get("answer")
    if result:
        if result["status"] == "abstain":
            st.warning("Insufficient policy evidence")
        st.text(result["answer"])
        for citation in result["citations"]:
            st.link_button(f'{citation["title"]} · {citation["location"]}', citation["url"])

with claims:
    pending = st.session_state.get("pending")
    if pending:
        st.info("Allow read access to this synthetic fixture for human review?")
        approve, decline = st.columns(2)
        decision = True if approve.button("Allow lookup") else None
        if decline.button("Decline lookup"):
            decision = False
        if decision is not None:
            with st.spinner("Recording decision"):
                review = request(f'/reviews/{pending["run_id"]}/decision',
                                 payload={"approved": decision})
            if review:
                st.session_state.review = review
                st.session_state.pending = None
                st.rerun()
        if st.button("Discard pending review"):
            st.session_state.pending = None
            st.rerun()
    else:
        with st.form("claim"):
            claim_id = st.text_input("Synthetic claim ID", placeholder="SYN-D-001")
            limit = st.number_input("Simulation limit (IDR)", min_value=1, max_value=10**12,
                                    value=None, step=1)
            policy_query = st.text_input("Optional policy question", max_chars=500)
            start = st.form_submit_button("Request review")
        if start:
            st.session_state.review = None
            pending = request("/reviews", payload={
                "claim_id": claim_id, "max_amount_idr": limit,
                "policy_query": policy_query.strip() or None,
            })
            if pending:
                st.session_state.pending = pending
                st.rerun()

    review = st.session_state.get("review")
    if review:
        result = review["result"]
        st.subheader({"review": "Ready for human review", "abstain": "Lookup abstained",
                      "escalate": "Needs human follow-up"}[result["status"]])
        st.text(result["summary"])
        if review["lookup"]:
            st.json(review["lookup"], expanded=False)
        for citation in result["citations"]:
            st.code(citation, language=None)
        if result["tool_calls"]:
            st.caption("Tools: " + ", ".join(result["tool_calls"]))

with history:
    st.button("Refresh run history")
    rows = request("/runs?limit=20")
    if rows:
        st.dataframe([{
            "Started (UTC)": item["started_at"], "Workflow": item["kind"].title(),
            "Outcome": item["status"], "Reason": item["reason"] or "",
            "Duration (ms)": item["duration_ms"], "Sources": item["citation_count"],
            "Tool calls": item["tool_call_count"], "Run ID": item["run_id"],
        } for item in rows], hide_index=True)
    elif rows == []:
        st.info("No completed runs yet.")
