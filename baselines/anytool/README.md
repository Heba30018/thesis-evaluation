# AnyTool Baseline (Python only)

A clean, standalone, Python-only implementation of the **AnyTool** baseline
(Du, Wei & Zhang, 2024 -- "AnyTool: Self-Reflective, Hierarchical Agents for
Large-Scale API Calls"), built on the same API pool (`endpoints.json`) used
in the earlier Django + C++ version of this project, but with no dependency
on Django, NetBox UI resolution, or C++. Same prompts, same hierarchical
structure, same self-reflection mechanism.

## Layout

```
anytool_baseline/
  endpoints.json                # the API pool (1167 endpoints, 11 categories, 138 tools)
  data/
    assigned_object_fields.json # optional hint for a handful of polymorphic fields
    ground_truth.json           # 500 test cases (see "Test data" below)
  anytool/
    data.py        # splits endpoints.json into category -> tool -> API
    schema.py       # reads each endpoint's request-body fields from the schema
    prompts.py       # every prompt (Meta/Category/Tool-Agent, their function
                      # schemas, the self-reflection reactivation messages, and
                      # the Solver with its small JSON-output adaptation)
    agent_loop.py     # the ReAct loop shared by every agent + exploration-call execution
    pipeline.py       # the full orchestrator: build hierarchy -> Solver -> Check -> Self-Reflection
    check.py          # deterministic validation of the Solver's chosen endpoint
    llm.py            # thin wrapper around OpenAI function calling
    evaluation.py      # comparison logic between a Solver answer and a ground-truth case
  run_query.py      # run a single query and print the result
  evaluate.py       # batch-test against data/ground_truth.json
```

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # put your OPENAI_API_KEY in there
python run_query.py "Create a new circuit type named 'Wireless Backup' with slug 'wireless-backup'."
python evaluate.py --limit 20      # quick dry run on 20 cases
python evaluate.py                  # all 500 cases (real OpenAI calls, costs money)
```

Each run's results are saved incrementally to `results/evaluation.json`, so
an interrupted run never loses progress. The terminal always prints every
agent's full prompt and function schema, every individual function call it
makes and what came back, its final input/output, and the Solver's own
prompt/input/output -- not hidden behind any flag -- so you can trace
exactly who called whom and what each step produced, for every case, not
just a one-line summary (see `HOW_IT_WORKS.md`).

There's too much of that to scroll back through in a terminal, so the
exact same text is also saved to a log file as it happens:
`results/run_query_trace.log` for `run_query.py`, and (by default)
`--output`'s path with a `.log` extension for `evaluate.py` --
`results/evaluation.log` for the default `--output`. Both accept
`--log-file` to pick a different path.

## Code map

| Concept | Here |
|---|---|
| Meta-Agent prompt | `prompts.build_meta_agent_prompt` |
| Category-Agent prompt | `prompts.build_category_agent_prompt` |
| Tool-Agent prompt | `prompts.build_tool_agent_prompt` |
| Each agent's functions | the functions inside each `build_*_agent_prompt` + `agent_loop.execute_tool_call` |
| Self-reflection reactivation messages | `prompts.tool_reactivation_message` / `category_reactivation_message` / `meta_reactivation_message` |
| Building/re-running one agent at a time | `pipeline.call_meta_agent` / `call_category_agent` / `call_tool_agent` (each always called individually, never merged) |
| Ascending reactivation order | `pipeline._self_reflection_round` (tool -> category -> meta) |
| The Solver + its "give up" option | `prompts.SOLVER_INSTRUCTIONS` + `pipeline._run_solver_stage` |
| Deterministic endpoint validation | `check.check_solver_output` |

## Test data

`data/ground_truth.json` is the same 500-case set used before, with one
adjustment for this offline setting: this baseline has no live NetBox to
resolve a name/slug into a numeric id, so for the 202 cases where the
expected endpoint originally carried an id that only a live lookup could
produce, that id has been replaced with the literal `"{id}"` placeholder --
exactly what the Solver itself returns in that situation (see
`prompts.SOLVER_INSTRUCTIONS`). The 48 cases where the task text states the
id outright (e.g. "Delete cable ID 65 ...") keep their real numeric id,
since the Solver can read it straight from the request. Either way, a
correct Solver answer matches the ground truth exactly -- no separate
lenient metric is needed.
