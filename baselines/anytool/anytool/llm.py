"""Thin OpenAI function-calling wrapper. Every agent and the Solver reach
the model only through this one function, so this is the only place the
OpenAI SDK is touched -- no fine-tuning or extra infrastructure is needed
beyond a normal chat-completions call with function-calling turned on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = os.environ.get("ANYTOOL_MODEL", "gpt-4o-mini")

_client: OpenAI | None = None

# One entry per call_chat() call that actually reached the API and got a
# usage back, so a script (evaluate.py, run_query.py) can report total
# token usage/cost for a whole run or a single query. See
# token_usage_totals(). A run through run_anytool() can make anywhere from
# a handful to dozens of these calls (Meta-/Category-/Tool-Agents, the
# Solver, and any self-reflection rounds), all going through this one
# function, so tracking it here -- rather than in pipeline.py -- catches
# every one of them automatically.
_usage: list[dict] = []


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    ok: bool
    content: str = ""
    raw_assistant_message: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str = ""


def call_chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None) -> ChatResult:
    """One OpenAI chat.completions call with optional function-calling
    tools. Never raises -- failures come back as ChatResult(ok=False)."""
    try:
        kwargs: dict[str, Any] = {"model": model or DEFAULT_MODEL, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = _get_client().chat.completions.create(**kwargs)
        message = response.choices[0].message

        if response.usage is not None:
            _usage.append({
                "model": kwargs["model"],
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            })

        tool_calls = []
        for tc in (message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return ChatResult(
            ok=True,
            content=message.content or "",
            raw_assistant_message=message.model_dump(exclude_none=True),
            tool_calls=tool_calls,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as ChatResult.error
        return ChatResult(ok=False, error=str(exc))


def token_usage_totals() -> dict:
    """Sums prompt/completion/total tokens across every call_chat() call
    made so far in this process -- e.g. every LLM call one evaluate.py or
    run_query.py run made, across the whole agent hierarchy and every
    self-reflection round."""
    return {
        "calls": len(_usage),
        "prompt_tokens": sum(u["prompt_tokens"] for u in _usage),
        "completion_tokens": sum(u["completion_tokens"] for u in _usage),
        "total_tokens": sum(u["total_tokens"] for u in _usage),
    }
