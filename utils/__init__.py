"""Shared utilities."""
import json


def parse_json_response(raw: str) -> str | list | dict:
    """Strip markdown fences and parse JSON from an LLM response."""
    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(cleaned)
