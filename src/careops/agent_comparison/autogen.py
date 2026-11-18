"""Run review and lookup actors through the local AutoGen message runtime."""

from dataclasses import dataclass

from autogen_core import (
    AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler,
)
from autogen_core.tools import FunctionTool

from careops.agent_comparison.task import CLAIM, POLICY, Review


@dataclass
class ReviewRequest:
    case_name: str


@dataclass
class LookupRequest:
    name: str
    value: str


@dataclass
class LookupReply:
    payload: dict


@dataclass
class ReviewReply:
    payload: dict


async def run(case, tools):
    review = Review(case)
    native = []

    def lookup_synthetic_claim(value: str) -> dict:
        return tools.dispatch(CLAIM, value)

    def lookup_policy(value: str) -> dict:
        return tools.dispatch(POLICY, value)

    functions = {
        CLAIM: FunctionTool(lookup_synthetic_claim, description="Read a synthetic claim."),
        POLICY: FunctionTool(lookup_policy, description="Resolve a local policy passage."),
    }

    class LookupAgent(RoutedAgent):
        @message_handler
        async def lookup(self, message: LookupRequest, ctx: MessageContext) -> LookupReply:
            native.append(message.name)
            result = await functions[message.name].run_json(
                {"value": message.value}, ctx.cancellation_token,
            )
            return LookupReply(result)

    class ReviewAgent(RoutedAgent):
        @message_handler
        async def handle(self, message: ReviewRequest, ctx: MessageContext) -> ReviewReply:
            if message.case_name != case.name:
                raise ValueError("Unknown review case")
            native.append("review")

            async def call(name, arg):
                reply = await self.send_message(
                    LookupRequest(name, arg), AgentId("lookup", self.id.key),
                    cancellation_token=ctx.cancellation_token,
                )
                return reply.payload

            if review.authorize() and await review.lookup_claim(call):
                await review.lookup_policy(call)
            return ReviewReply({"result": review.result, "lookup": review.claim})

    runtime = SingleThreadedAgentRuntime()
    await LookupAgent.register(runtime, "lookup", lambda: LookupAgent("Read-only lookup"))
    await ReviewAgent.register(runtime, "review", lambda: ReviewAgent("Bounded synthetic review"))
    runtime.start()
    try:
        reply = await runtime.send_message(ReviewRequest(case.name), AgentId("review", "local"))
    finally:
        await runtime.stop_when_idle()
    return reply.payload, native
