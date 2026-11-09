"""Turn any Python function into a Runnable."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from triage.chains import clean_ticket
from triage.data import ACCOUNT_TICKET
from triage.model import get_model
from triage.prompts import CATEGORY_PROMPT

messy_ticket = f"   {ACCOUNT_TICKET}\n\n   "

# A wrapped function gets invoke, batch, stream and | like any other Runnable.
clean = RunnableLambda(clean_ticket)
print("Cleaned:", clean.invoke(messy_ticket))
print("Batch:  ", clean.batch(["  a   b ", "write to me@example.org."]))

# Lambda functions take exactly one input, so build a dict for the prompt.
to_prompt_input = RunnableLambda(lambda ticket: {"ticket": ticket})
chain = clean | to_prompt_input | CATEGORY_PROMPT | get_model() | StrOutputParser()
print("Chain:  ", chain.invoke(messy_ticket))

# Next to another Runnable, a plain function is wrapped for you.
shorter = clean | (lambda ticket: ticket[:25] + "...")
print("Auto:   ", shorter.invoke(messy_ticket))
