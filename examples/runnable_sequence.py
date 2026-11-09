"""Everything is a Runnable, and | builds a RunnableSequence."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableSequence

from triage.data import CRASH_TICKET
from triage.model import get_model
from triage.prompts import CATEGORY_PROMPT

model = get_model()
parser = StrOutputParser()

for component in (CATEGORY_PROMPT, model, parser):
    print(f"{type(component).__name__} is a Runnable: {isinstance(component, Runnable)}")

# These two are equivalent.
piped = CATEGORY_PROMPT | model | parser
explicit = RunnableSequence(CATEGORY_PROMPT, model, parser)

print("\nType of the piped chain:", type(piped).__name__)
print("Steps:", [type(step).__name__ for step in piped.steps])

inputs = {"ticket": CRASH_TICKET}
print("\nPiped:   ", piped.invoke(inputs))
print("Explicit:", explicit.invoke(inputs))

# invoke on a sequence is just this loop.
print("\nStep by step:")
value = inputs
for step in piped.steps:
    value = step.invoke(value)
    print(f"  {type(step).__name__} -> {type(value).__name__}")
