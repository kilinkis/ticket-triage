"""The triage pipeline and the chains it is built from."""

import re
from operator import itemgetter

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, chain

from triage.model import get_model
from triage.prompts import CATEGORY_PROMPT, KEYWORDS_PROMPT, REPLY_PROMPT, SENTIMENT_PROMPT

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def clean_ticket(text: str) -> str:
    """Collapse whitespace and mask email addresses before text reaches a model."""
    return _EMAIL.sub("[email]", " ".join(text.split()))


def prepare_ticket(raw_ticket: str) -> dict:
    return {"ticket": clean_ticket(raw_ticket)}


def normalize_label(text: str) -> str:
    """Real models sometimes answer 'Billing.' when asked for 'billing'."""
    return text.strip().strip(".").lower()


def needs_escalation(analysis: dict) -> bool:
    return analysis["sentiment"] == "negative" and analysis["category"] in {"billing", "technical"}


def build_category_chain(model: BaseChatModel | None = None) -> Runnable:
    # A plain function next to a Runnable is wrapped in a RunnableLambda automatically.
    return CATEGORY_PROMPT | (model or get_model()) | StrOutputParser() | normalize_label


def build_sentiment_chain(model: BaseChatModel | None = None) -> Runnable:
    return SENTIMENT_PROMPT | (model or get_model()) | StrOutputParser() | normalize_label


def build_keywords_chain(model: BaseChatModel | None = None) -> Runnable:
    return KEYWORDS_PROMPT | (model or get_model()) | CommaSeparatedListOutputParser()


def build_reply_chain(model: BaseChatModel | None = None) -> Runnable:
    return REPLY_PROMPT | (model or get_model()) | StrOutputParser()


def build_analysis_chain(model: BaseChatModel | None = None) -> Runnable:
    """Turn {"ticket": ...} into the ticket plus category, sentiment and keywords."""
    model = model or get_model()
    return RunnableParallel(
        # Each branch returns only its own result, so the ticket has to be copied
        # through explicitly or the reply step won't have it.
        ticket=itemgetter("ticket"),
        category=build_category_chain(model),
        sentiment=build_sentiment_chain(model),
        keywords=build_keywords_chain(model),
    )


def build_triage_pipeline(model: BaseChatModel | None = None) -> Runnable:
    """Raw ticket text in, analysis plus an action and optional reply out.

    Data shape at each step:
        str
        -> {"ticket"}                                     prepare_ticket
        -> {"ticket", "category", "sentiment", "keywords"} analysis (in parallel)
        -> the same keys + {"action", "reply"}             route_ticket
    """
    model = model or get_model()
    reply_chain = build_reply_chain(model)

    # Defined inside this function so it uses the reply_chain built from the same model.
    @chain
    def route_ticket(analysis: dict) -> dict:
        if needs_escalation(analysis):
            return {**analysis, "action": "escalate", "reply": None}
        return {**analysis, "action": "auto_reply", "reply": reply_chain.invoke(analysis)}

    return RunnableLambda(prepare_ticket) | build_analysis_chain(model) | route_ticket
