"""Pipe a prompt, a model and an output parser into a chain."""

from langchain_core.output_parsers import StrOutputParser

from triage.data import BILLING_TICKET
from triage.model import get_model
from triage.prompts import CATEGORY_PROMPT

# The output of each step becomes the input of the next.
category_chain = CATEGORY_PROMPT | get_model() | StrOutputParser()

print("Ticket:  ", BILLING_TICKET)
print("Category:", category_chain.invoke({"ticket": BILLING_TICKET}))
