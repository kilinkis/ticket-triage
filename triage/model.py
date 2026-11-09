"""The one place that decides which chat model the project uses."""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from triage.fake_model import FakeTriageChatModel

load_dotenv()


def get_model(
    temperature: float = 0.0,
    max_tokens: int | None = None,
    fake_latency: float = 0.0,
) -> BaseChatModel:
    """Return a chat model chosen by the LLM_PROVIDER environment variable.

    LLM_PROVIDER=fake (the default) needs no key and makes no network calls.
    LLM_PROVIDER=openai uses OPENAI_API_KEY and OPENAI_MODEL.
    fake_latency only applies to the fake model.
    """
    provider = os.getenv("LLM_PROVIDER", "fake").lower()

    if provider == "fake":
        return FakeTriageChatModel(
            temperature=temperature, max_tokens=max_tokens, latency=fake_latency
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        options = {} if max_tokens is None else {"max_tokens": max_tokens}
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            **options,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'fake' or 'openai'.")
