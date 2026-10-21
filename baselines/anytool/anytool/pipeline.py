
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import llm, prompts
from .agent_loop import AgentInstance, run_agent
from .check import check_solver_output

SEPARATOR = "*" * 80


@dataclass
class RunState:
    meta: AgentInstance = field(default_factory=AgentInstance)
    categories: list[str] = field(default_factory=list)

    category_agents: dict[str, AgentInstance] = field(default_factory=dict)
    tools_by_category: dict[str, list[str]] = field(default_factory=dict)

    tool_agents: dict[str, AgentInstance] = field(default_factory=dict)
    endpoints_by_tool: dict[str, list[str]] = field(default_factory=dict)

    solver_messages: list[dict] = field(default_factory=list)
    solver_candidate_key_map: dict[str, str] = field(default_factory=dict)


    agent_stats: dict[str, dict] = field(
        default_factory=lambda: {
            role: {"seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
            for role in ("meta", "category", "tool", "solver")
        }
    )


@dataclass
class SolveResult:
    ok: bool
    answer: dict[str, Any] | None = None
    error: str = ""
    rounds_used: int = 0
    candidate_pool: list[str] = field(default_factory=list)
    # Copy of the RunState.agent_stats this run finished with -- see its
    # docstring. Always has all four keys, even for a role that never ran
    # (its value just stays all-zero).
    agent_stats: dict[str, dict] = field(default_factory=dict)


def _format_call_stats(tag: str, call_stats: dict) -> str:
    """One trace line for a single agent/Solver call's own time and token
    usage -- printed right after that call's own OUT line."""
    return (
        f"{tag} TIME={call_stats['seconds']:.2f}s  "
        f"TOKENS: {call_stats['prompt_tokens']} in / {call_stats['completion_tokens']} out"
    )


# ---------------------------------------------------------------------------
# Collect helpers: identical logic whether an agent runs for the first time
# or is resumed by self-reflection, so both paths append to the SAME list.
# ---------------------------------------------------------------------------

def _collect_unique(value: str, out: list[str]) -> None:
    if value and value not in out:
        out.append(value)


def _collect_category(args: dict, out: list[str]) -> None:
    name = args.get("category_name")
    if isinstance(name, str):
        _collect_unique(name, out)


def _collect_tools(args: dict, out: list[str]) -> None:
    for t in args.get("tool_list") or []:
        if isinstance(t, str):
            _collect_unique(t, out)


def _collect_apis(args: dict, context: dict, out: list[str]) -> None:
    for e in args.get("APIs") or []:
        if isinstance(e, str):
            _collect_unique(e, out)
    context["candidate_pool"] = out  # so a later check_if_request_solvable sees the true running pool


# ---------------------------------------------------------------------------
# Hierarchy construction (idempotent: each agent is created once, then only
# ever re-activated). Each function below is responsible for exactly one
# agent and is always called on its own -- nothing here merges several
# agents into one call, so each step is traceable by itself.
# ---------------------------------------------------------------------------

def call_meta_agent(state: RunState, query: str, model: str) -> None:
    if state.meta.messages:
        return
    built = prompts.build_meta_agent_prompt(query)
    state.meta.messages = [{"role": "user", "content": built["prompt"]}]
    state.meta.functions = built["functions"]
    state.meta.decision_function = "create_agent_category_level"
    state.meta.log_tag = "[Meta-Agent]"


    usage_before = llm.token_usage_totals()
    started = time.time()
    run_agent(state.meta, model, lambda args, ctx: _collect_category(args, state.categories))
    usage_after = llm.token_usage_totals()
    call_stats = {
        "seconds": time.time() - started,
        "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
        "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
    }
    totals = state.agent_stats["meta"]
    totals["seconds"] += call_stats["seconds"]
    totals["prompt_tokens"] += call_stats["prompt_tokens"]
    totals["completion_tokens"] += call_stats["completion_tokens"]

    # print(f"{state.meta.log_tag} OUT categories={state.categories} finished={state.meta.finished}", flush=True)
    # print(_format_call_stats(state.meta.log_tag, call_stats), flush=True)
    # print(SEPARATOR, flush=True)

def call_category_agent(state: RunState, query: str, category: str, model: str) -> None:
    if category in state.category_agents:
        return

    try:
        built = prompts.build_category_agent_prompt(query, category)
    except ValueError as exc:
        print(f"[skip] {exc}", flush=True)
        return

    agent = AgentInstance(
        messages=[{"role": "user", "content": built["prompt"]}],
        functions=built["functions"],
        decision_function="create_agent_tool_level",
        log_tag=f"[Category-Agent:{category}]",
        context={"category": category},
    )

    usage_before = llm.token_usage_totals()
    started = time.time()
    try:
        run_agent(
            agent,
            model,
            lambda args, ctx: _collect_tools(
                args, state.tools_by_category.setdefault(category, [])
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- one failing agent must not stop the run
        print(f"{agent.log_tag} failed: {exc}", flush=True)
    usage_after = llm.token_usage_totals()

    call_stats = {
        "seconds": time.time() - started,
        "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
        "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
    }
    totals = state.agent_stats["category"]
    totals["seconds"] += call_stats["seconds"]
    totals["prompt_tokens"] += call_stats["prompt_tokens"]
    totals["completion_tokens"] += call_stats["completion_tokens"]

    state.category_agents[category] = agent


def call_tool_agent(state: RunState, query: str, category: str, tool: str, model: str) -> None:
    """Runs one Tool-Agent once. Input: the user's query, the category it
    belongs to, and the one tool (or small group of tools) the
    Category-Agent assigned it. Output: the endpoint_ids it decided are
    relevant (state.endpoints_by_tool[category|tool]), which feed directly
    into the Solver's candidate pool. Calling this again for a
    (category, tool) pair that already has an agent does nothing."""
    key = f"{category}|{tool}"
    if key in state.tool_agents:
        return
    built = prompts.build_tool_agent_prompt(query, category, tool)
    agent = AgentInstance(
        messages=[{"role": "user", "content": built["prompt"]}],
        functions=built["functions"],
        decision_function="add_API_into_API_pool",
        log_tag=f"[Tool-Agent:{category}.{tool}]",
        context={"category": category, "candidate_pool": []},
    )

    # print(SEPARATOR, flush=True)
    # print(f"{agent.log_tag} IN  query={query!r} category={category!r} tool={tool!r}", flush=True)
    # print(f"{agent.log_tag} PROMPT:\n{built['prompt']}", flush=True)

    usage_before = llm.token_usage_totals()
    started = time.time()
    run_agent(agent, model, lambda args, ctx: _collect_apis(args, ctx, state.endpoints_by_tool.setdefault(key, [])))
    usage_after = llm.token_usage_totals()
    call_stats = {
        "seconds": time.time() - started,
        "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
        "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
    }
    totals = state.agent_stats["tool"]
    totals["seconds"] += call_stats["seconds"]
    totals["prompt_tokens"] += call_stats["prompt_tokens"]
    totals["completion_tokens"] += call_stats["completion_tokens"]

    state.tool_agents[key] = agent
    # print(f"{agent.log_tag} OUT endpoint_ids={state.endpoints_by_tool.get(key, [])} finished={agent.finished}", flush=True)
    # print(_format_call_stats(agent.log_tag, call_stats), flush=True)
    # print(SEPARATOR, flush=True)


def _self_reflection_round(state: RunState, query: str, model: str, fail_reason: str) -> None:
   
    for key, agent in state.tool_agents.items():
        if agent.finished:
            continue
        msg = prompts.tool_reactivation_message(fail_reason)
        agent.messages.append({"role": "user", "content": msg})
        print(SEPARATOR, flush=True)
        print(f"{agent.log_tag} IN  (reactivation) reason={fail_reason!r}", flush=True)
        print(f"{agent.log_tag} REACTIVATION MESSAGE:\n{msg}", flush=True)

        usage_before = llm.token_usage_totals()
        started = time.time()
        run_agent(
            agent, model,
            lambda args, ctx, key=key: _collect_apis(args, ctx, state.endpoints_by_tool.setdefault(key, [])),
        )
        usage_after = llm.token_usage_totals()
        call_stats = {
            "seconds": time.time() - started,
            "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
            "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
        }
        totals = state.agent_stats["tool"]
        totals["seconds"] += call_stats["seconds"]
        totals["prompt_tokens"] += call_stats["prompt_tokens"]
        totals["completion_tokens"] += call_stats["completion_tokens"]

        # print(f"{agent.log_tag} OUT endpoint_ids={state.endpoints_by_tool.get(key, [])} finished={agent.finished}", flush=True)
        # print(_format_call_stats(agent.log_tag, call_stats), flush=True)
        # print(SEPARATOR, flush=True)

    for category, agent in state.category_agents.items():
        if agent.finished:
            continue
        msg = prompts.category_reactivation_message(fail_reason)
        agent.messages.append({"role": "user", "content": msg})
        # print(SEPARATOR, flush=True)
        # print(f"{agent.log_tag} IN  (reactivation) reason={fail_reason!r}", flush=True)
        # print(f"{agent.log_tag} REACTIVATION MESSAGE:\n{msg}", flush=True)

        usage_before = llm.token_usage_totals()
        started = time.time()
        run_agent(
            agent, model,
            lambda args, ctx, category=category: _collect_tools(args, state.tools_by_category.setdefault(category, [])),
        )
        usage_after = llm.token_usage_totals()
        call_stats = {
            "seconds": time.time() - started,
            "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
            "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
        }
        totals = state.agent_stats["category"]
        totals["seconds"] += call_stats["seconds"]
        totals["prompt_tokens"] += call_stats["prompt_tokens"]
        totals["completion_tokens"] += call_stats["completion_tokens"]

        # print(f"{agent.log_tag} OUT tools={state.tools_by_category.get(category, [])} finished={agent.finished}", flush=True)
        # print(_format_call_stats(agent.log_tag, call_stats), flush=True)
        # print(SEPARATOR, flush=True)

    if not state.meta.finished:
        msg = prompts.meta_reactivation_message(fail_reason)
        state.meta.messages.append({"role": "user", "content": msg})
        # print(SEPARATOR, flush=True)
        # print(f"{state.meta.log_tag} IN  (reactivation) reason={fail_reason!r}", flush=True)
        # print(f"{state.meta.log_tag} REACTIVATION MESSAGE:\n{msg}", flush=True)

        usage_before = llm.token_usage_totals()
        started = time.time()
        run_agent(state.meta, model, lambda args, ctx: _collect_category(args, state.categories))
        usage_after = llm.token_usage_totals()
        call_stats = {
            "seconds": time.time() - started,
            "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
            "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
        }
        totals = state.agent_stats["meta"]
        totals["seconds"] += call_stats["seconds"]
        totals["prompt_tokens"] += call_stats["prompt_tokens"]
        totals["completion_tokens"] += call_stats["completion_tokens"]

        # print(f"{state.meta.log_tag} OUT categories={state.categories} finished={state.meta.finished}", flush=True)
        # print(_format_call_stats(state.meta.log_tag, call_stats), flush=True)
        # print(SEPARATOR, flush=True)

    # Any category/tool named for the first time just now (by a
    # reactivated Meta- or Category-Agent) still needs its own agent,
    # called individually, exactly like the first build of the hierarchy.
    for category in list(state.categories):
        call_category_agent(state, query, category, model)
    for category, tools in list(state.tools_by_category.items()):
        for tool in tools:
            call_tool_agent(state, query, category, tool, model)


def _current_pool(state: RunState, pruned: set[str]) -> list[str]:
    """Deduplicated union of every Tool-Agent's collected endpoint_ids,
    minus whatever the Solver has, in a prior round, named as irrelevant."""
    pool, seen = [], set()
    for endpoint_ids in state.endpoints_by_tool.values():
        for eid in endpoint_ids:
            if eid in pruned or eid in seen:
                continue
            seen.add(eid)
            pool.append(eid)
    return pool


def _all_agents_finished(state: RunState) -> bool:
    """True once every agent in the hierarchy has called "finish_search" --
    at that point self-reflection has nothing left to re-activate, so
    another round could only help by discovering something new, which
    `run_anytool` checks for separately (the candidate pool growing)."""
    return (
        state.meta.finished
        and all(a.finished for a in state.category_agents.values())
        and all(a.finished for a in state.tool_agents.values())
    )


# ---------------------------------------------------------------------------
# Solver stage
# ---------------------------------------------------------------------------

def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _run_solver_stage(state: RunState, query: str, model: str, is_reactivation: bool, pool: list[str]) -> dict:
    """Returns {"ok", "answer" | "error", "irrelevant_endpoint_keys"}."""
    print(SEPARATOR, flush=True)
    print(f"[Solver] IN  query={query!r} candidate_pool={pool} is_reactivation={is_reactivation}", flush=True)
    if not is_reactivation:
        built = prompts.build_solver_prompt(query, pool)
        state.solver_messages = [{"role": "user", "content": built["prompt"]}]
        state.solver_candidate_key_map = built["candidate_key_map"]
        print(f"[Solver] PROMPT:\n{built['prompt']}", flush=True)
    else:
        built = prompts.build_solver_reactivation_prompt(pool)
        state.solver_messages.append({"role": "user", "content": built["message"]})
        state.solver_candidate_key_map = built["candidate_key_map"]
        print(f"[Solver] REACTIVATION MESSAGE:\n{built['message']}", flush=True)

    usage_before = llm.token_usage_totals()
    started = time.time()
    result = llm.call_chat(state.solver_messages, model=model)
    usage_after = llm.token_usage_totals()
    call_stats = {
        "seconds": time.time() - started,
        "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
        "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
    }
    totals = state.agent_stats["solver"]
    totals["seconds"] += call_stats["seconds"]
    totals["prompt_tokens"] += call_stats["prompt_tokens"]
    totals["completion_tokens"] += call_stats["completion_tokens"]
    # Printed once here, right after the call -- not repeated after each of
    # the several "[Solver] OUT ..." lines below, since exactly one of
    # those always follows right after this regardless of which branch the
    # response falls into.
    # print(_format_call_stats("[Solver]", call_stats), flush=True)
    if not result.ok:
        print(f"[Solver] OUT OpenAI call failed: {result.error}", flush=True)
        print(SEPARATOR, flush=True)
        return {"ok": False, "error": f"Solver OpenAI call failed: {result.error}", "irrelevant_endpoint_keys": []}

    print(f"[Solver] RAW OUTPUT (exactly what the LLM returned this call): {result.content!r}", flush=True)

    state.solver_messages.append(result.raw_assistant_message)

    if not result.content.strip():
        print("[Solver] OUT empty response", flush=True)
        print(SEPARATOR, flush=True)
        return {"ok": False, "error": "Solver returned no output.", "irrelevant_endpoint_keys": []}

    try:
        answer = _parse_json_object(result.content)
    except json.JSONDecodeError as exc:
        print(f"[Solver] OUT not valid JSON: {exc}", flush=True)
        print(SEPARATOR, flush=True)
        return {"ok": False, "error": f"Solver output is not JSON: {exc}", "irrelevant_endpoint_keys": []}

    if isinstance(answer, dict) and isinstance(answer.get("error"), str):
        print(f"[Solver] OUT gave up: {answer['error']}", flush=True)
        print(SEPARATOR, flush=True)
        return {
            "ok": False,
            "error": answer["error"],
            "irrelevant_endpoint_keys": [k for k in (answer.get("irrelevant_endpoints") or []) if isinstance(k, str)],
        }

    # Deterministic check: validate the chosen (method, endpoint) against
    # endpoints.json directly, rather than trusting the model's own claim
    # that the endpoint exists.
    method, endpoint = answer.get("method", ""), answer.get("endpoint", "")
    if not method or not endpoint:
        print(f"[Solver] OUT missing method/endpoint: {answer}", flush=True)
        print(SEPARATOR, flush=True)
        return {"ok": False, "error": "Solver output is missing method/endpoint.", "irrelevant_endpoint_keys": []}

    check = check_solver_output(method, endpoint)
    if not check["valid"]:
        print(f"[Solver] OUT failed schema check: {method} {endpoint}", flush=True)
        print(SEPARATOR, flush=True)
        return {"ok": False, "error": "Solver's answer failed Check (schema validation).", "irrelevant_endpoint_keys": []}

    print(f"[Solver] OUT answer={answer}", flush=True)
    print(SEPARATOR, flush=True)
    return {"ok": True, "answer": answer, "irrelevant_endpoint_keys": []}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_anytool(query: str, model: str | None = None) -> SolveResult:
    model = model or llm.DEFAULT_MODEL

    state = RunState()
    pruned: set[str] = set()
    fail_reason = ""
    previous_pool_size: int | None = None

    print(SEPARATOR, flush=True)
    print(f"[run_anytool] IN  query={query!r} model={model!r}", flush=True)
    print(SEPARATOR, flush=True)

    call_meta_agent(state, query, model)
    for category in list(state.categories):
        call_category_agent(state, query, category, model)
    for category, tools in list(state.tools_by_category.items()):
        for tool in tools:
            call_tool_agent(state, query, category, tool, model)

    round_index = 0
    while True:
        if round_index > 0:
            _self_reflection_round(state, query, model, fail_reason)

        pool = _current_pool(state, pruned)
        if not pool:
            fail_reason = (
                "AnyTool found no candidate endpoint across any category/tool "
                + ("(first pass)." if round_index == 0 else "(after self-reflection retry).")
            )
        else:
            stage = _run_solver_stage(state, query, model, is_reactivation=round_index > 0, pool=pool)

            if stage["ok"]:
                print(SEPARATOR, flush=True)
                print(f"[run_anytool] OUT ok=True answer={stage['answer']} rounds_used={round_index}", flush=True)
                print(SEPARATOR, flush=True)
                return SolveResult(
                    ok=True, answer=stage["answer"], rounds_used=round_index, candidate_pool=pool,
                    agent_stats={role: dict(stats) for role, stats in state.agent_stats.items()},
                )

            fail_reason = stage["error"]

            if stage["irrelevant_endpoint_keys"]:
                inverse = {v: k for k, v in state.solver_candidate_key_map.items()}
                for key in stage["irrelevant_endpoint_keys"]:
                    if key in inverse:
                        pruned.add(inverse[key])

        # Natural termination condition, not a fixed round count: once a
        # round changes nothing at all -- the candidate pool didn't grow,
        # and every agent has already called "finish_search" so there is
        # no one left to re-activate -- there is nothing left for another
        # round to try.
        if round_index > 0 and len(pool) == previous_pool_size and _all_agents_finished(state):
            break
        previous_pool_size = len(pool)
        round_index += 1

    print(SEPARATOR, flush=True)
    print(f"[run_anytool] OUT ok=False error={fail_reason!r} rounds_used={round_index}", flush=True)
    print(SEPARATOR, flush=True)
    return SolveResult(
        ok=False, error=fail_reason, rounds_used=round_index,
        agent_stats={role: dict(stats) for role, stats in state.agent_stats.items()},
    )
