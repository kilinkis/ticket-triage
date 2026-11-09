"""Feed the output of a RunnableParallel into the next Runnable."""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from triage.data import BILLING_TICKET
from triage.model import get_model
from triage.prompts import CATEGORY_PROMPT, REPLY_PROMPT, SENTIMENT_PROMPT

model = get_model()

# The parallel step outputs exactly the keys REPLY_PROMPT needs.
analysis = RunnableParallel(
    ticket=itemgetter("ticket"),
    category=CATEGORY_PROMPT | model | StrOutputParser(),
    sentiment=SENTIMENT_PROMPT | model | StrOutputParser(),
)
reply_pipeline = analysis | REPLY_PROMPT | model | StrOutputParser()

print("Parallel step:", analysis.invoke({"ticket": BILLING_TICKET}))
print("\nDraft reply:  ", reply_pipeline.invoke({"ticket": BILLING_TICKET}))

# Inside a pipe, a plain dict is automatically turned into a RunnableParallel.
shorthand = (
    {
        "ticket": itemgetter("ticket"),
        "category": CATEGORY_PROMPT | model | StrOutputParser(),
        "sentiment": SENTIMENT_PROMPT | model | StrOutputParser(),
    }
    | REPLY_PROMPT
    | model
    | StrOutputParser()
)
print("\nDict shorthand:", shorthand.invoke({"ticket": BILLING_TICKET}))
