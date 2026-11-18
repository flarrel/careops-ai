"""Run the review as a Semantic Kernel plugin calling two registered tool functions."""

from semantic_kernel import Kernel
from semantic_kernel.filters import FilterTypes
from semantic_kernel.functions import KernelArguments, kernel_function

from careops.agent_comparison.task import CLAIM, POLICY, Review


async def run(case, tools):
    kernel = Kernel()
    review = Review(case)
    native = []

    @kernel.filter(FilterTypes.FUNCTION_INVOCATION)
    async def observe(context, next):
        await next(context)
        native.append(context.function.name)

    class Lookups:
        @kernel_function(name=CLAIM, description="Read one synthetic claim.")
        def claim(self, value: str) -> dict:
            return tools.dispatch(CLAIM, value)

        @kernel_function(name=POLICY, description="Resolve one local policy passage.")
        def policy(self, value: str) -> dict:
            return tools.dispatch(POLICY, value)

    async def call(name, arg):
        result = await kernel.invoke(
            function_name=name, plugin_name="read", arguments=KernelArguments(value=arg)
        )
        return result.value

    class Workflow:
        @kernel_function(name="review", description="Run the approved fixture review.")
        async def run_review(self) -> dict:
            if review.authorize() and await review.lookup_claim(call):
                await review.lookup_policy(call)
            return {"result": review.result, "lookup": review.claim}

    kernel.add_plugin(Lookups(), "read")
    kernel.add_plugin(Workflow(), "workflow")
    reply = await kernel.invoke(function_name="review", plugin_name="workflow")
    return reply.value, native
