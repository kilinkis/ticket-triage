"""Output parsers turn a model's message into a useful Python value."""

from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser

from triage.data import BILLING_TICKET
from triage.model import get_model
from triage.parsers import IsoDateOutputParser
from triage.prompts import CATEGORY_PROMPT, DATE_PROMPT, KEYWORDS_PROMPT

model = get_model()
ticket = {"ticket": BILLING_TICKET}

# Without a parser you get an AIMessage object.
message = model.invoke(CATEGORY_PROMPT.invoke(ticket))
print("Raw message:", repr(message.content), f"({type(message).__name__})")

# StrOutputParser extracts the text.
print("String:     ", repr(StrOutputParser().invoke(message)))

# Parsers can tell the model what format to use. That text goes in the prompt.
list_parser = CommaSeparatedListOutputParser()
print("\nFormat instructions:", list_parser.get_format_instructions())
print("List:       ", list_parser.invoke(model.invoke(KEYWORDS_PROMPT.invoke(ticket))))

# A custom parser only needs a parse() method.
date_parser = IsoDateOutputParser()
parsed_date = date_parser.invoke(model.invoke(DATE_PROMPT.invoke(ticket)))
print("Date:       ", repr(parsed_date))
