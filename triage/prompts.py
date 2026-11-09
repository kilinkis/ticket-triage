"""Prompt templates for each triage step."""

from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

from triage.parsers import IsoDateOutputParser

CATEGORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You triage customer support tickets. Classify the ticket into exactly one "
            "category: billing, technical, account, or other. Reply with the category only.",
        ),
        ("human", "{ticket}"),
    ]
)

SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You read customer support tickets. Label the customer's sentiment as positive, "
            "neutral, or negative. Reply with one word only.",
        ),
        ("human", "{ticket}"),
    ]
)

KEYWORDS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract keywords from customer support tickets. Return at most three "
            "keywords. {format_instructions}",
        ),
        ("human", "{ticket}"),
    ]
).partial(format_instructions=CommaSeparatedListOutputParser().get_format_instructions())

DATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract dates from customer support tickets. Find the date the problem "
            "happened. {format_instructions}",
        ),
        ("human", "{ticket}"),
    ]
).partial(format_instructions=IsoDateOutputParser().get_format_instructions())

REPLY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly support agent. Write a short reply to the customer, "
            "matching your tone to their sentiment.",
        ),
        ("human", "Category: {category}\nSentiment: {sentiment}\nTicket: {ticket}"),
    ]
)
