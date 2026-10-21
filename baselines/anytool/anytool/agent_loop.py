"""The ReAct loop shared by every agent in the hierarchy (Meta, Category,
Tool): send the running conversation + its function list to the model,
execute whatever it calls, feed the result back, repeat until IT decides
to call "finish_search" -- that is the only stopping rule: nothing here
second-guesses the agent or cuts it off early.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import data as d
from . import llm

# Every individual function call an agent makes -- its name, the exact
# arguments it was called with, and the real data that came back -- is
# always printed directly to the terminal (not hidden behind any flag),
# right under the agent's own IN/PROMPT/FUNCTIONS block printed in
# pipeline.py. This is what lets you see, concretely, what a Tool-Agent
# got back when it called e.g. get_APIs_in_tool -- the real endpoint list,
# not just the function's schema.


@dataclass
class AgentInstance:
    """One running agent (Meta-, a Category-, or a Tool-Agent): its own
    conversation, function list, decision function (the call that records
    its output - e.g. "add_API_into_API_pool" for a Tool-Agent), and
    whether it has reached "Finished" status.

    Kept alive for the whole run: an agent only becomes "Finished" once it
    calls "finish_search" itself, and a "Finished" agent is skipped by
    self-reflection, so a later self-reflection round can re-activate a
    not-yet-finished agent by appending a new turn instead of starting
    over.
    """

    messages: list[dict] = field(default_factory=list)
    functions: list[dict] = field(default_factory=list)
    decision_function: str = ""
    log_tag: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    finished: bool = False


def execute_tool_call(function_name: str, arguments: dict, context: dict) -> dict:
    """Server-side execution of one exploration/self-check call -- every
    function an agent can call other than its own decision function and
    "finish_search", both of which are handled directly in the loop below.
    `context` carries whatever is fixed for this agent instance (its own
    category, its own running candidate pool for
    "check_if_request_solvable")."""
    category = context.get("category")
    pool = context.get("candidate_pool") or []

    if function_name == "get_tools_in_category":
        category_name = category or arguments.get("category_name")
        return {"tools": d.get_tools_in_category(category_name)}

    if function_name == "get_tool_descriptions":
        category_name = category or arguments.get("category_name")
        tool_list = arguments.get("tool_list") or []
        return {"tool_descriptions": d.get_tool_descriptions(category_name, tool_list)}

    if function_name == "get_APIs_in_tool":
        tool_name = arguments.get("tool_name")
        return {"apis": d.get_APIs_in_tool(category, tool_name)}

    if function_name == "get_API_detail":
        return {"api_details": d.get_API_detail(arguments.get("APIs") or [])}

    if function_name == "check_if_request_solvable":
        # A quick, cheap check against THIS agent's own pool so far (not
        # the global pool, which only exists once the whole hierarchy
        # finishes): true iff it is non-empty and resolves to real APIs.
        if not pool:
            return {"solvable": False, "reason": "candidate pool is still empty"}
        resolvable = d.keys_for_endpoint_ids(pool)
        return {"solvable": len(resolvable) > 0, "checked_count": len(pool), "resolvable_count": len(resolvable)}

    return {"error": f"unknown function '{function_name}'"}


def run_agent(agent: AgentInstance, model: str, collect: Callable[[dict, dict], None]) -> None:

    turn = 0
    while True:
        turn += 1
        print(f"{agent.log_tag} TURN {turn} PROMPT (every message being sent to the LLM this call):", flush=True)
        for i, msg in enumerate(agent.messages):
            print(f"  [{i}] {msg}", flush=True)

        usage_before = llm.token_usage_totals()
        result = llm.call_chat(agent.messages, agent.functions, model)
        if not result.ok:
            print(f"{agent.log_tag} OpenAI call failed: {result.error}", flush=True)
            return
        usage_after = llm.token_usage_totals()
        turn_prompt_tokens = usage_after["prompt_tokens"] - usage_before["prompt_tokens"]
        turn_completion_tokens = usage_after["completion_tokens"] - usage_before["completion_tokens"]

        print(
            f"{agent.log_tag} TURN {turn}  "
            f"{turn_prompt_tokens} in / {turn_completion_tokens} out",
            flush=True,
        )
        print(f"{agent.log_tag} TURN {turn} RAW OUTPUT ", flush=True)
        print(f"  {result.raw_assistant_message}", flush=True)

        agent.messages.append(result.raw_assistant_message)

        if not result.tool_calls:
            return  # plain-text reply, nothing more to do

        finished_this_turn = False
        tool_responses = []

        for call in result.tool_calls:
            if call.name == agent.decision_function:
                collect(call.arguments, agent.context)
                observation = {"status": "recorded"}
            elif call.name == "finish_search":
                finished_this_turn = True
                observation = {"status": "ok"}
            else:
                observation = execute_tool_call(call.name, call.arguments, agent.context)

            print(f"{agent.log_tag} called {call.name}({call.arguments}) -> {observation}", flush=True)
            tool_responses.append({"role": "tool", "tool_call_id": call.id, "content": _dump(observation)})

        if finished_this_turn:
            agent.finished = True
            return

        agent.messages.extend(tool_responses)


def _dump(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
