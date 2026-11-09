"""Run a chain over many inputs, optionally at the same time."""

import time

from langchain_core.output_parsers import StrOutputParser

from triage.data import SAMPLE_TICKETS
from triage.model import get_model
from triage.prompts import CATEGORY_PROMPT

# fake_latency makes each call slow, so the concurrency difference is visible.
category_chain = CATEGORY_PROMPT | get_model(fake_latency=0.3) | StrOutputParser()
inputs = [{"ticket": ticket} for ticket in SAMPLE_TICKETS]

for max_concurrency in (1, 5):
    start = time.perf_counter()
    results = category_chain.batch(inputs, config={"max_concurrency": max_concurrency})
    print(f"max_concurrency={max_concurrency}: {time.perf_counter() - start:.2f}s")

# Results always come back in the same order as the inputs.
print()
for ticket, category in zip(SAMPLE_TICKETS, results):
    print(f"{category:<10} {ticket[:60]}")
