# How the AnyTool baseline works

A step-by-step walkthrough of the code, in the order it actually executes.

## The core idea, and how it differs from Gorilla

AnyTool is agentic, hierarchical, and allowed to retry. A query is handed
to a **Meta-Agent**, which assigns it to one or more **Category-Agents**,
each of which assigns it to one or more **Tool-Agents**; the Tool-Agents
collect a shared pool of candidate API endpoints; a single **Solver** then
tries to answer the query using only that pool. If the Solver fails, a
**Self-Reflection** pass re-activates the whole hierarchy to gather better
candidates, and the Solver tries again — for as long as another round
could plausibly help, exactly like the paper describes it. Every
agent uses OpenAI function-calling in a small ReAct loop (reason, call a
function, observe the result, repeat) rather than a single forward pass,
which is the fundamental architectural difference from the Gorilla
baseline sitting next to this project.

## 1) `anytool/data.py` — the category → tool → API hierarchy

The paper organizes its API pool into three tiers (Section 4.1): category,
tool, and API. This file builds that tree from `endpoints.json` by
splitting each entry's `endpoint_id` (shaped like
`"<category>.<tool>.<name>"`, e.g. `"circuits.circuit-types.list"`) on its
dots — the dataset already carries its own taxonomy, so no separate file is
needed.

`raw_endpoints()` loads the file once and caches it; `_load()` builds two
lookup shapes from it: a `category -> tool -> [records]` tree, and a flat
`endpoint_id -> record` map, also cached. Both are protected by a
`threading.RLock()` (not a plain `Lock`, since `_load()` calls
`raw_endpoints()` while already holding the lock — a plain lock would
deadlock there), because the paper runs the hierarchy's agents "in a
multi-threaded manner" and they all share this same cached data.

On top of that tree sit the functions each tier's agent actually calls:
`get_categories()` / `get_tools_in_category()` / `get_tool_descriptions()`
for the Meta- and Category-Agents, and `get_apis()` / `get_APIs_in_tool()` /
`get_API_detail()` for the Tool-Agent — these three Tool-Agent-facing names
match, on purpose, the function names the Tool-Agent itself sees in its own
function-calling schema in `prompts.py`. A last group —
`keys_for_endpoint_ids()` and `candidate_key_map_for_endpoint_ids()` —
turns a list of collected `endpoint_id`s into the `"METHOD /path"` strings
the Solver's prompt is actually built from.

## 2) `anytool/schema.py` — reading real request-body fields

When the Solver's prompt lists a candidate endpoint, it needs that
endpoint's *real* request-body fields, not guessed ones. `find_endpoint_schema`
looks up an endpoint's raw `endpoints.json` entry by (method, path), and
`format_candidate_block` formats one candidate's block in the Solver's
prompt: its path, its required/optional fields, and — for a handful of
polymorphic NetBox fields (`assigned_object_type`/`assigned_object_id`) — an
optional hint loaded from `data/assigned_object_fields.json`, if that file
is present.

## 3) `anytool/prompts.py` — every prompt and function schema

This is the part that matters most for staying faithful to the paper.
Nothing in this file talks to OpenAI or walks the hierarchy — it only
builds text and function schemas for `agent_loop.py` and `pipeline.py` to
use.

- `build_meta_agent_prompt` (paper Table 14): tells the model it is
  "APIGPT" with access to a database organized into categories, and gives
  it three functions — `get_tools_in_category`, `get_tool_descriptions`,
  and the decision function `create_agent_category_level` (assign a
  category to a new Category-Agent) — plus `finish_search`.
- `build_category_agent_prompt` (Table 15): same idea, one tier down —
  this agent's decision function is `create_agent_tool_level`, which
  assigns a batch of at most five tools to a new Tool-Agent.
- `build_tool_agent_prompt` (Table 16): the bottom tier. Its functions are
  `get_APIs_in_tool`, `get_API_detail`, `check_if_request_solvable`, the
  decision function `add_API_into_API_pool` (this is the actual Python
  code behind that function name you asked about earlier — it lives here,
  not in a separate file, as the `functions` list entry inside this
  builder), and `finish_search`.
- `tool_reactivation_message` / `category_reactivation_message` /
  `meta_reactivation_message` (Tables 18, 19, 20): the exact messages
  appended as a new turn when self-reflection re-activates an agent that
  hasn't finished yet. They don't rebuild a fresh prompt — the agent keeps
  its own memory of what it already explored.
- `SOLVER_INSTRUCTIONS` + `build_solver_prompt` (adapted from Table 17,
  via ToolLLM): a single-shot prompt — not a ReAct loop — that lists every
  candidate endpoint the hierarchy collected (with its real fields from
  `schema.py`), and asks the model to pick one and build a request, or
  give up with `{"error": "..."}` and optionally name which candidates it's
  sure are irrelevant (so self-reflection can prune them instead of
  re-offering the same wrong ones).
- `build_solver_reactivation_prompt`: the Solver's own self-reflection —
  appends the updated candidate pool as a new turn in its existing
  conversation, rather than starting over.

## 4) `anytool/agent_loop.py` — the shared ReAct loop

Every agent in the hierarchy (Meta, Category, Tool) runs through the same
loop, because they all follow the same pattern: send the conversation and
function list to the model, execute whatever function it calls, feed the
result back as a new turn, repeat.

`AgentInstance` is the dataclass holding one running agent's state: its
`messages`, its `functions`, which function is its *decision* function
(the one that actually records output — e.g. `add_API_into_API_pool` for a
Tool-Agent), a `log_tag` for diagnostics, a `context` dict (e.g. which
category this agent owns), and a `finished` flag. Agents are kept alive
for the whole run rather than thrown away, specifically so a later
self-reflection round can resume them with a new turn instead of starting
from scratch — this matches the paper's rule that only agents *not* marked
"Finished" (i.e., haven't called `finish_search`) get re-activated.

`execute_tool_call()` is the server-side implementation of every
*non*-decision function a model might call — `get_tools_in_category`,
`get_tool_descriptions`, `get_APIs_in_tool`, `get_API_detail`, and
`check_if_request_solvable` (which checks whether this agent's own running
pool, so far, is non-empty and resolves to real APIs).

`run_agent(agent, model, collect)` is the loop itself: each
iteration calls the model, then for every function call the model made:
if it's the agent's decision function, call `collect(arguments,
agent.context)` (this is what actually grows the hierarchy's shared lists
— e.g. appends a new category name, or a new `endpoint_id`, to the right
list in `pipeline.py`'s state); if it's `finish_search`, mark the agent
`finished` and stop; otherwise, execute it via `execute_tool_call` and
feed the observation back. The loop keeps going, with no bound at all,
until the agent itself calls `finish_search`, with nothing here cutting
it off early. The only other way out is an OpenAI call itself failing (a
dropped connection, say), which is always printed directly (no flag
needed) and ends that activation without marking the agent `finished` —
a real, if rare, edge
case: if a call keeps failing on every retry (a sustained outage), that
one agent would never report itself done, and self-reflection would keep
re-activating it indefinitely, since there is no turn or round cap left
anywhere to catch that case. That's the one gap left by removing both
caps entirely; everything else behaves exactly as the paper describes.

## 5) `anytool/pipeline.py` — the full orchestrator

This is where everything above is assembled into one run, via
`run_anytool(query, model)`.

**Building the hierarchy.** `call_meta_agent` builds and runs the
Meta-Agent once (idempotent — a second call does nothing), collecting
category names into `state.categories`, and logs what went in and what
came out (`[Meta-Agent] IN ...` / `OUT ...`). `call_category_agent` /
`call_tool_agent` do the same one tier down, each keyed by name (or
`category|tool`) so an agent is only ever created once and later only
re-activated, and each logs its own input/output the same way. There is
no single function that builds the whole hierarchy at once — every new
agent is always called on its own, by name. Both `run_anytool` (right
after the Meta-Agent's first pass) and `_self_reflection_round` (after
every self-reflection round) contain the same two small loops: one that
calls `call_category_agent` for every category named so far, and one that
calls `call_tool_agent` for every (category, tool) pair named so far —
each loop only actually creates an agent for a category/tool that doesn't
have one yet, so re-running the loop after a reactivation only picks up
what's new.

**The candidate pool.** `_current_pool` is the deduplicated union of every
Tool-Agent's collected `endpoint_id`s, minus whatever the Solver has, in an
earlier round, named as irrelevant (`pruned`) — this is the pool the Solver
actually sees.

**The Solver stage.** `_run_solver_stage` builds the Solver's prompt (fresh
the first time, or a reactivation turn on a retry) from the current pool,
calls the model once, and parses its JSON reply. If the reply names
`"error"`, that's the Solver giving up — any `irrelevant_endpoints` it
listed get translated back from `"METHOD /path"` to `endpoint_id` and
pruned from the pool for next time. Otherwise, the chosen (method,
endpoint) goes through a deterministic check (`check.check_solver_output`)
that verifies it against `endpoints.json` directly rather than trusting
the model — a numeric id in the path is first folded back to `{id}` since
the schema is keyed by the templated path, not a specific instance.
`_run_solver_stage` also logs its own input (the query and the candidate
pool it's being given) and output (the final answer, or exactly why it
failed) the same way every agent does.

**Self-reflection.** `_self_reflection_round` is the retry mechanism:
every agent that hasn't finished gets re-activated, **in ascending
order** — every unfinished Tool-Agent first, then every unfinished
Category-Agent, then the Meta-Agent last — each with its own matching
reactivation message appended as a new turn (not a fresh prompt), and each
logged the same way as a first-time call. After that, the same two
"call every category/tool named so far" loops described above run again,
so anything new those reactivations named gets its own agent. The ascending order matters: a Tool-Agent gets first
crack at finding more APIs within tools it already knows about, and only
if that's not enough does the problem escalate up to a Category-Agent (to
assign more tools) and then the Meta-Agent (to assign more categories
entirely).

**The run loop.** `run_anytool` ties it together: build the hierarchy
once, then keep looping. Round 0 tries the Solver against whatever pool
the first pass produced. Each later round first calls
`_self_reflection_round` (reactivating agents and growing the pool), then
tries the Solver again. The loop returns as soon as the Solver succeeds.
If it doesn't, the loop keeps going for as long as another round could
plausibly help — the paper's own words for self-reflection are that it
"can be applied repeatedly until the termination condition is met," with
no fixed round count, so there is no fixed round count here either.
Concretely, after each round `_all_agents_finished(state)` checks whether
every agent (Meta, every Category-Agent, every Tool-Agent) has already
called `finish_search` — those are exactly the agents `_self_reflection_round`
would skip re-activating next time, per the paper's own rule that
"Finished" agents are excluded from self-reflection. If that's true *and*
the candidate pool didn't grow this round either, there is nothing left
for another round to discover, and that — not a round counter — is the
real stopping condition, with no numeric cap underneath it at all.

Two different loops are easy to confuse here: the ReAct loop inside
`agent_loop.run_agent` runs *inside* each single agent
activation, calling the model repeatedly until that one agent calls
`finish_search`; the round loop inside `run_anytool` wraps a whole cascade
of those agent activations (Meta, then every Category- and Tool-Agent)
and repeats *that* for as long as self-reflection can still plausibly
help. Both are now bounded only by their own real stopping condition —
the agent's own `finish_search` call, and the hierarchy actually running
out of things to try — exactly matching the paper, which places no
numeric limit on either one. The one trade-off that comes with removing
both caps entirely is the edge case described in section 4 above: an
OpenAI call that keeps failing (not the model deciding anything, just a
dropped connection or a sustained outage) leaves that one agent forever
not-"Finished," so self-reflection would keep re-activating it with no
cap left to catch that case.

## 6) `anytool/check.py` — deterministic validation

`canonicalize()` folds a numeric id in a path (e.g. `/7/`) back to
`{id}`. `check_solver_output(method, endpoint)` looks up that canonical
(method, endpoint) pair directly in `endpoints.json` via
`schema.find_endpoint_schema`: trust the dataset, not the model, to decide
whether the Solver's chosen endpoint is real.

## 7) `anytool/llm.py` — calling the model

A thin wrapper around one `chat.completions.create` call with optional
function-calling `tools`. `call_chat()` never raises; any failure comes
back as `ChatResult(ok=False, error=...)`. Every agent and the Solver both
go through this same function — it's the only place the OpenAI SDK is
touched.

## 8) `anytool/evaluation.py` — comparing against the ground truth

`evaluate_case(expected_answer, actual)` compares the Solver's parsed
answer against one `ground_truth.json` case on three things:
`method_match`, `endpoint_match` (after normalizing trailing slashes and
query parameters), and `body_match` (exact field-by-field comparison).
`correct` is true only when all three match — no separate lenient
"template-only" fallback is needed, because `data/ground_truth.json`
already uses the literal `"{id}"` placeholder in exactly the cases where
the Solver itself would have to answer with `{id}` (see the project
README's "Test data" section).

## 9) `run_query.py` and `evaluate.py` — running it

`run_query.py` runs one query through `run_anytool()` and prints the
number of self-reflection rounds used, the final candidate pool, and the
answer (or the error, if it failed). `evaluate.py` loops over
`data/ground_truth.json` (optionally sliced with `--limit`), running
`run_anytool()` on each case's task text, comparing with `evaluate_case()`,
printing each case's own trace between two `"-----"` markers ending in
`[i/n] CASE_ID: correct` / `incorrect`, and saving progress to
`results/evaluation.json` after every single case, so an interrupted run
never loses what it already did. Neither script has a verbosity flag --
every agent's input and output (what query, category, or tool went in;
what categories/tools/endpoint_ids came out), its full prompt and
function schema, every individual function call it made along the way,
and the Solver's own prompt/input/output are always printed, so you can
always trace exactly who called whom and what each step produced.

In short, the flow for every query is: `pipeline.run_anytool` builds the
Meta → Category → Tool hierarchy (`agent_loop.py` running each agent's
ReAct loop, `data.py` + `prompts.py` supplying what each one can see and
call) → collects a shared candidate pool → the Solver
(`pipeline._run_solver_stage` + `prompts.SOLVER_INSTRUCTIONS`) tries to
answer from that pool → `check.py` validates the answer deterministically
→ on failure, self-reflection (`pipeline._self_reflection_round`)
re-activates the hierarchy in ascending order for another attempt, for as
many rounds as the hierarchy keeps finding something new to try →
`evaluation.py` + `evaluate.py` compare the final answer against the
ground truth and record the result.
