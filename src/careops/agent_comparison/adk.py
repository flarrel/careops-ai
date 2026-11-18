"""Run approval, claim and policy as ADK agents with session-state routing."""

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types

from careops.agent_comparison.task import CLAIM, POLICY, Review


async def run(case, tools):
    review = Review(case)
    native = []

    def lookup_synthetic_claim(value: str) -> dict:
        """Read one synthetic claim."""
        return tools.dispatch(CLAIM, value)

    def lookup_policy(value: str) -> dict:
        """Resolve one local policy passage."""
        return tools.dispatch(POLICY, value)

    functions = {CLAIM: FunctionTool(lookup_synthetic_claim), POLICY: FunctionTool(lookup_policy)}

    class Stage(BaseAgent):
        async def _run_async_impl(self, ctx):
            calls = []

            async def call(name, arg):
                calls.append(name)
                return await functions[name].run_async(
                    args={"value": arg}, tool_context=ToolContext(ctx)
                )

            changes = {}
            if self.name == "approval":
                changes["authorized"] = review.authorize()
            elif self.name == "claim" and ctx.session.state.get("authorized"):
                changes["policy_ready"] = await review.lookup_claim(call)
            elif self.name == "policy" and ctx.session.state.get("policy_ready"):
                await review.lookup_policy(call)
            for name in calls:
                yield Event(invocation_id=ctx.invocation_id, author=self.name,
                            custom_metadata={"tool_name": name})
            yield Event(
                invocation_id=ctx.invocation_id, author=self.name,
                actions=EventActions(state_delta=changes),
            )

    root = SequentialAgent(name="review", sub_agents=[
        Stage(name="approval"), Stage(name="claim"), Stage(name="policy"),
    ])
    runner = InMemoryRunner(app_name="careops_lab", agent=root)
    try:
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="local-benchmark",
        )
        async for event in runner.run_async(
            user_id=session.user_id, session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="Run fixture review.")]),
        ):
            native.append((event.custom_metadata or {}).get("tool_name", event.author))
    finally:
        await runner.close()
    return {"result": review.result, "lookup": review.claim}, native
