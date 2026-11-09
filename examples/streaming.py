"""Receive the answer piece by piece as it is generated."""

from langchain_core.output_parsers import StrOutputParser

from triage.data import CRASH_TICKET
from triage.model import get_model
from triage.prompts import REPLY_PROMPT

reply_chain = REPLY_PROMPT | get_model(fake_latency=0.05) | StrOutputParser()

inputs = {"ticket": CRASH_TICKET, "category": "technical", "sentiment": "neutral"}
for chunk in reply_chain.stream(inputs):
    print(chunk, end="|", flush=True)  # The | shows where each chunk ends.
print()
