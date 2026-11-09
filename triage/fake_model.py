"""An offline chat model, so the repo runs without an API key.

It subclasses BaseChatModel, which means invoke, batch, stream and the | operator
are LangChain's real code. Only the text is made up: simple keyword rules choose
an answer based on the system prompt and the ticket.

The answer depends only on the prompt, never on call order, so results stay
correct under batch and RunnableParallel concurrency.
"""

import re
import time
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_CATEGORY_WORDS = {
    "billing": ("charged", "refund", "invoice", "payment", "subscription"),
    "technical": ("crash", "error", "bug", "broken"),
    "account": ("password", "account", "username", "login"),
}
_NEGATIVE_WORDS = ("unacceptable", "frustrated", "angry", "terrible", "worst")
_POSITIVE_WORDS = ("thanks", "thank you", "great", "love")
_STOPWORDS = {
    "about", "again", "have", "keeps", "really", "there", "their", "this",
    "when", "with", "would", "your", "currently", "change", "offer",
}


def _classify(ticket: str) -> str:
    text = ticket.lower()
    for category, words in _CATEGORY_WORDS.items():
        if any(word in text for word in words):
            return category
    return "other"


def _sentiment(ticket: str) -> str:
    text = ticket.lower()
    if any(word in text for word in _NEGATIVE_WORDS):
        return "negative"
    if any(word in text for word in _POSITIVE_WORDS):
        return "positive"
    return "neutral"


def _keywords(ticket: str) -> str:
    words = re.findall(r"[a-z]{5,}", ticket.lower())
    unique = [word for word in dict.fromkeys(words) if word not in _STOPWORDS]
    return ", ".join(unique[:3])


def _date(ticket: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", ticket)
    return match.group() if match else "none"


def _reply(ticket: str) -> str:
    category = re.search(r"^Category:\s*(\w+)", ticket, re.MULTILINE)
    sentiment = re.search(r"^Sentiment:\s*(\w+)", ticket, re.MULTILINE)
    category_name = category.group(1) if category else _classify(ticket)
    sentiment_name = sentiment.group(1) if sentiment else _sentiment(ticket)

    topic = "your question" if category_name == "other" else f"your {category_name} issue"
    apology = "I am really sorry for the trouble. " if sentiment_name == "negative" else ""
    return (
        f"Hi there, thanks for contacting us about {topic}. {apology}"
        "Our team will follow up within one business day."
    )


# Checked in order against the lowercased system prompt. The reply prompt also
# mentions sentiment, so it has to be matched first.
_TASKS = {
    "reply to the customer": _reply,
    "classify the ticket": _classify,
    "sentiment": _sentiment,
    "keywords": _keywords,
    "dates": _date,
}


class FakeTriageChatModel(BaseChatModel):
    """Answers triage prompts with keyword rules instead of calling an API."""

    latency: float = 0.0
    """Seconds to sleep per call (or per chunk when streaming), to make timing visible."""
    temperature: float = 0.0
    """Accepted to match a real model. It has no effect here."""
    max_tokens: int | None = None
    """Truncates the answer. Words stand in for tokens."""

    @property
    def _llm_type(self) -> str:
        return "fake-triage"

    def _respond(self, messages: list[BaseMessage]) -> str:
        system = " ".join(
            str(message.content) for message in messages if isinstance(message, SystemMessage)
        ).lower()
        humans = [message for message in messages if isinstance(message, HumanMessage)]
        ticket = str(humans[-1].content) if humans else ""

        text = next(
            (task(ticket) for marker, task in _TASKS.items() if marker in system),
            f"(fake model) You said: {ticket}",
        )
        if self.max_tokens is not None:
            text = " ".join(text.split()[: self.max_tokens])
        return text

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        time.sleep(self.latency)
        message = AIMessage(content=self._respond(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for token in re.findall(r"\S+\s*", self._respond(messages)):
            time.sleep(self.latency)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
