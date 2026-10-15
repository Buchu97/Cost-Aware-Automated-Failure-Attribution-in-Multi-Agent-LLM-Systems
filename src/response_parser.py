"""Parse structured LLM attribution responses into individual fields."""

import re
from typing import Optional


def parse_attribution_response(text: str) -> dict:
    """Parse the LLM's structured response into a dict.

    Expected format::

        Agent Name: <name>
        Step Number: <number>
        Confidence: <0-100>
        Reason: <explanation>

    Returns:
        Dict with keys: agent_name, step_number, confidence, reason.
        Values are None when parsing fails for that field.
    """
    result: dict[str, Optional[str | int]] = {
        "agent_name": None,
        "step_number": None,
        "confidence": None,
        "reason": None,
    }

    if not text:
        return result

    # Agent Name — handles **Agent Name:**, Agent Name:, etc.
    m = re.search(r"Agent\s*Name[\s\*]*:[\s\*]*(.+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).split("\n")[0].strip()
        # Strip trailing/leading quotes, markdown stars, backticks
        cleaned = raw.strip("\"'*` ")
        result["agent_name"] = cleaned

    # Step Number
    m = re.search(r"Step\s*Number[\s\*]*:[\s\*]*(\d+)", text, re.IGNORECASE)
    if m:
        result["step_number"] = m.group(1)

    # Confidence
    m = re.search(r"Confidence[\s\*]*:[\s\*]*(\d+)", text, re.IGNORECASE)
    if m:
        conf = int(m.group(1))
        result["confidence"] = min(max(conf, 0), 100)  # clamp 0-100

    # Reason — everything after 'Reason:' until end
    m = re.search(r"Reason[\s\*]*:[\s\*]*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        result["reason"] = m.group(1).strip().strip("\"'*` ")

    return result
