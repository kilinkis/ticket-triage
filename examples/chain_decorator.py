"""@chain turns a whole function into a Runnable.

It behaves like RunnableLambda. It's handy when a step needs real Python logic,
like this if statement, and calls other chains inside.
"""

from langchain_core.runnables import RunnableLambda, chain

from triage.chains import build_reply_chain, build_triage_pipeline, needs_escalation
from triage.data import ACCOUNT_TICKET, BILLING_TICKET, SAMPLE_TICKETS
from triage.model import get_model

reply_chain = build_reply_chain(get_model())


@chain
def route_ticket(analysis: dict) -> str:
    if needs_escalation(analysis):
        return "ESCALATED to a human agent"
    return reply_chain.invoke(analysis)


print("route_ticket is a:", type(route_ticket).__name__)
assert isinstance(route_ticket, RunnableLambda)

print("\nUnhappy billing customer:")
print(" ", route_ticket.invoke({"ticket": BILLING_TICKET, "category": "billing", "sentiment": "negative"}))
print("Happy account customer:")
print(" ", route_ticket.invoke({"ticket": ACCOUNT_TICKET, "category": "account", "sentiment": "positive"}))

# The same routing inside the full pipeline (see triage/chains.py).
print("\nFull pipeline:\n")
pipeline = build_triage_pipeline()
for result in pipeline.batch(SAMPLE_TICKETS):
    print(f"[{result['action']}] {result['category']} / {result['sentiment']} / {result['keywords']}")
    print(f"  Ticket: {result['ticket']}")
    if result["reply"]:
        print(f"  Reply:  {result['reply']}")
    print()
