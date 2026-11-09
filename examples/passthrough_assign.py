"""Pipe chains together and carry earlier values forward."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from triage.data import BILLING_TICKET
from triage.model import get_model
from triage.prompts import CATEGORY_PROMPT, REPLY_PROMPT, SENTIMENT_PROMPT

model = get_model()
category_chain = CATEGORY_PROMPT | model | StrOutputParser()
sentiment_chain = SENTIMENT_PROMPT | model | StrOutputParser()
reply_chain = REPLY_PROMPT | model | StrOutputParser()

# RunnablePassthrough returns its input untouched.
print("Passthrough:", RunnablePassthrough().invoke({"anything": 1}))

# category_chain outputs only a string, so piping it straight into reply_chain
# would lose the ticket. assign adds new keys and keeps the existing ones.
pipeline = (
    RunnablePassthrough.assign(category=category_chain)
    | RunnablePassthrough.assign(sentiment=sentiment_chain)
    | RunnablePassthrough.assign(reply=reply_chain)
)

result = pipeline.invoke({"ticket": BILLING_TICKET})
for key, value in result.items():
    print(f"{key:<10} {value}")
