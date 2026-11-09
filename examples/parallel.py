"""Run independent chains at the same time on the same input."""

import time

from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel

from triage.data import BILLING_TICKET
from triage.model import get_model
from triage.prompts import CATEGORY_PROMPT, KEYWORDS_PROMPT, SENTIMENT_PROMPT

model = get_model(fake_latency=0.3)

parallel = RunnableParallel(
    category=CATEGORY_PROMPT | model | StrOutputParser(),
    sentiment=SENTIMENT_PROMPT | model | StrOutputParser(),
    keywords=KEYWORDS_PROMPT | model | CommaSeparatedListOutputParser(),
)

start = time.perf_counter()
result = parallel.invoke({"ticket": BILLING_TICKET})
elapsed = time.perf_counter() - start

print(result)
print(f"\nThree 0.3s calls took {elapsed:.2f}s, not 0.9s.")
