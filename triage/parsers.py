"""A custom output parser.

langchain-core 1.x has no DatetimeOutputParser. A parser only needs parse(),
plus get_format_instructions() if the prompt should describe the format.
"""

from datetime import date

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser


class IsoDateOutputParser(BaseOutputParser[date | None]):
    """Parse a YYYY-MM-DD answer into a date, or 'none' into None."""

    def parse(self, text: str) -> date | None:
        cleaned = text.strip().strip(".").lower()
        if cleaned == "none":
            return None
        try:
            return date.fromisoformat(cleaned)
        except ValueError as error:
            raise OutputParserException(f"Expected YYYY-MM-DD or 'none', got: {text!r}") from error

    def get_format_instructions(self) -> str:
        return "Answer with a single date formatted as YYYY-MM-DD, or the word none."
