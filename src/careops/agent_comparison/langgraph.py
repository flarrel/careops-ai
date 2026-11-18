"""Run the maintained graph with the common comparison tools."""

from careops.agent import ReviewAgent


async def run(case, tools):
    events = []
    agent = ReviewAgent(tools.claim, tools.policy, events.append, budget=case.budget)
    review = agent.start(case.task().model_dump())
    response = review.resume(case.approved)
    native = [event.get("tool_name", event["stage"]) for event in events]
    return response, native
