from __future__ import annotations

import json
import os

from openai import OpenAI

_client: OpenAI | None = None

# One entry per complete() call made so far in this process, so a script
# can report total token usage/cost for a whole run. See token_usage_totals().
_usage: list[dict] = []


def client() -> OpenAI:
    """Returns a shared OpenAI client, built the first time it's needed."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def complete(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """Sends one prompt as a user message and returns the model's reply text."""
    messages = [{"role": "user", "content": prompt}]

    response = client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    if response.usage is not None:
        _usage.append({
            "model": model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        })
    return (response.choices[0].message.content or "").strip()


def token_usage_totals() -> dict:
    return {
        "calls": len(_usage),
        "prompt_tokens": sum(u["prompt_tokens"] for u in _usage),
        "completion_tokens": sum(u["completion_tokens"] for u in _usage),
        "total_tokens": sum(u["total_tokens"] for u in _usage),
    }
