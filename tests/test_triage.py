import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
from langchain_core.exceptions import OutputParserException

from triage.chains import (
    build_analysis_chain,
    build_category_chain,
    build_reply_chain,
    build_triage_pipeline,
    clean_ticket,
)
from triage.data import (
    ACCOUNT_TICKET,
    ANGRY_BUG_TICKET,
    BILLING_TICKET,
    CRASH_TICKET,
    QUESTION_TICKET,
    SAMPLE_TICKETS,
)
from triage.fake_model import FakeTriageChatModel
from triage.model import get_model
from triage.parsers import IsoDateOutputParser

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")


@pytest.mark.parametrize(
    ("ticket", "category"),
    [
        (BILLING_TICKET, "billing"),
        (CRASH_TICKET, "technical"),
        (ACCOUNT_TICKET, "account"),
        (QUESTION_TICKET, "other"),
    ],
)
def test_category_chain_classifies_tickets(ticket, category):
    assert build_category_chain().invoke({"ticket": ticket}) == category


def test_batch_keeps_input_order_under_concurrency():
    tickets = [{"ticket": ticket} for ticket in SAMPLE_TICKETS]
    model = FakeTriageChatModel(latency=0.01)
    results = build_category_chain(model).batch(tickets, config={"max_concurrency": 5})
    assert results == ["billing", "technical", "account", "other", "technical"]


def test_stream_yields_chunks_that_join_to_the_full_answer():
    chain = build_reply_chain()
    inputs = {"ticket": CRASH_TICKET, "category": "technical", "sentiment": "neutral"}
    chunks = list(chain.stream(inputs))
    assert len(chunks) > 1
    assert "".join(chunks) == chain.invoke(inputs)


def test_analysis_returns_ticket_and_all_fields():
    result = build_analysis_chain().invoke({"ticket": BILLING_TICKET})
    assert result["ticket"] == BILLING_TICKET
    assert result["category"] == "billing"
    assert result["sentiment"] == "negative"
    assert isinstance(result["keywords"], list) and result["keywords"]


@pytest.mark.parametrize(
    ("ticket", "action"),
    [
        (BILLING_TICKET, "escalate"),
        (ANGRY_BUG_TICKET, "escalate"),
        (CRASH_TICKET, "auto_reply"),
        (ACCOUNT_TICKET, "auto_reply"),
    ],
)
def test_pipeline_routes_tickets(ticket, action):
    result = build_triage_pipeline().invoke(ticket)
    assert result["action"] == action
    assert (result["reply"] is None) == (action == "escalate")


def test_pipeline_masks_email_before_analysis():
    result = build_triage_pipeline().invoke(ACCOUNT_TICKET)
    assert "alex@example.com" not in result["ticket"]
    assert "[email]" in result["ticket"]


def test_clean_ticket_keeps_the_sentence_period():
    assert clean_ticket("  mail   me@example.co.uk. ") == "mail [email]."


def test_date_parser():
    parser = IsoDateOutputParser()
    assert parser.parse("2026-11-03") == date(2026, 11, 3)
    assert parser.parse("none") is None
    with pytest.raises(OutputParserException):
        parser.parse("last Tuesday")


def test_default_provider_is_fake(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER")
    assert isinstance(get_model(), FakeTriageChatModel)


def test_openai_provider_builds_chat_openai_without_calling_it(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = get_model(temperature=0.5, max_tokens=5)
    assert type(model).__name__ == "ChatOpenAI"
    assert model.temperature == 0.5


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ValueError, match="nope"):
        get_model()


def test_examples_are_found():
    assert EXAMPLES


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_example_runs(script):
    env = {**os.environ, "LLM_PROVIDER": "fake"}
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
