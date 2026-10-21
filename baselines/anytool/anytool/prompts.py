"""Every prompt and function schema AnyTool's agents use:

  - Meta-Agent / Category-Agent / Tool-Agent bootstrap prompts: the text
    each agent first sees, telling it what it is and what it can do.
  - Their function lists (name, purpose, input, output): the OpenAI
    function-calling schemas each agent is given alongside its prompt.
  - Self-reflection re-activation messages: short follow-up messages
    appended to an agent's own conversation when it is asked to try again.
  - The Solver: a prompt adapted so the model returns one structured JSON
    action instead of free text, since this API pool's "solution" is a
    concrete API call (method + endpoint + request body), not a prose
    answer.

Nothing here talks to OpenAI or walks the hierarchy -- see agent_loop.py
and pipeline.py for that. This module only builds the text and function
schemas each step is asked to reason over.
"""

from __future__ import annotations

from typing import Any

from . import data as d
from .schema import find_endpoint_schema, format_candidate_block


# ===========================================================================
# Tier 1 -- Meta-Agent
# ===========================================================================

def build_meta_agent_prompt(user_query: str) -> dict[str, Any]:
    categories = d.get_categories()
    if not categories:
        raise ValueError("No categories found in endpoints.json.")
    names = [c["category"] for c in categories]

    prompt = f"""You are APIGPT, with access to a database of APIs. This database is
organized into the following categories: {", ".join(names)}. Your task is
to help users identify the relevant categories for their needs. To do
this, you can use the 'get_tools_in_category' function to retrieve the
available tools within a specific category. If you are unsure about the
functionality of some tools, the 'get_tool_descriptions' function can be
used to obtain detailed information about these tools. This information
will aid you in understanding the general functionality of each category.
Additionally, the 'create_agent_category_level' function allows you to
assign a relevant category to an agent, with each agent being assigned
only one category. However, you can assign multiple categories to
different agents. It is important to explore as many categories as
possible, as the solution to a query may be found in unexpected
categories. Remember, your goal is not to answer the query directly but
to identify all potentially relevant categories and assign them to
agents. Once you have completed the assignment, call the 'finish_search'
function. At each step, you should briefly analyze the current status and
determine your next action, including the function calls needed to
execute your step. Keep your analysis concise, ideally no longer than
three sentences.

User request:
{user_query}"""

    functions = [
        {"type": "function", "function": {
            "name": "get_tools_in_category",
            "description": "Get the tools available under one category.",
            "parameters": {"type": "object", "properties": {
                "category_name": {"type": "string", "enum": names},
            }, "required": ["category_name"]},
        }},
        {"type": "function", "function": {
            "name": "get_tool_descriptions",
            "description": "Get a short description of one or more tools within a category.",
            "parameters": {"type": "object", "properties": {
                "category_name": {"type": "string", "enum": names},
                "tool_list": {"type": "array", "items": {"type": "string"}},
            }, "required": ["category_name", "tool_list"]},
        }},
        {"type": "function", "function": {
            "name": "create_agent_category_level",
            "description": "Assign one relevant category to a new Category-Agent. Call once per category.",
            "parameters": {"type": "object", "properties": {
                "category_name": {"type": "string", "enum": names},
            }, "required": ["category_name"]},
        }},
        {"type": "function", "function": {
            "name": "finish_search",
            "description": "Call once every relevant category has been assigned.",
            "parameters": {"type": "object", "properties": {}},
        }},
    ]
    return {"prompt": prompt, "functions": functions}


# ===========================================================================
# Tier 2 -- Category-Agent
# ===========================================================================

def build_category_agent_prompt(user_query: str, category: str) -> dict[str, Any]:
    tools = d.get_tools_in_category(category)
    if not tools:
        raise ValueError(f"Unknown category '{category}'.")
    names = [t["tool"] for t in tools]

    prompt = f"""You are APIGPT, with access to a database of APIs categorized into
various groups. Each category contains numerous tools, and each tool
encompasses multiple APIs. Your task is to assist users in finding
relevant tools within the "{category}" category. If uncertain about the
functionality of some tools, use the 'get_tool_descriptions' function to
obtain detailed information. Then, employ the 'create_agent_tool_level'
function to allocate a subset of pertinent tools to an agent, ensuring
that similar tools are assigned to the same agent and limiting the
allocation to no more than five tools per agent. Remember, your role is
not to answer queries directly, but to assign all possible tools. Once
you complete the assignment, or if you determine the query is irrelevant
to the tools in the specified category, invoke the 'finish_search'
function. Execute each step by calling the appropriate functions, and
keep your thought process concise, ideally within three sentences.

User request:
{user_query}"""

    functions = [
        {"type": "function", "function": {
            "name": "get_tools_in_category",
            "description": f"List the tools available in '{category}' (this agent's only category).",
            "parameters": {"type": "object", "properties": {
                "category_name": {"type": "string", "enum": [category]},
            }, "required": ["category_name"]},
        }},
        {"type": "function", "function": {
            "name": "get_tool_descriptions",
            "description": f"Get a short description for one or more tools within '{category}'.",
            "parameters": {"type": "object", "properties": {
                "category_name": {"type": "string", "enum": [category]},
                "tool_list": {"type": "array", "items": {"type": "string", "enum": names}},
            }, "required": ["category_name", "tool_list"]},
        }},
        {"type": "function", "function": {
            "name": "create_agent_tool_level",
            "description": f"Assign a Tool-Agent a subset (at most 5) of tools from '{category}'. Call again for another subset.",
            "parameters": {"type": "object", "properties": {
                "tool_list": {"type": "array", "items": {"type": "string", "enum": names}, "maxItems": 5},
            }, "required": ["tool_list"]},
        }},
        {"type": "function", "function": {
            "name": "finish_search",
            "description": "Call once every possible tool has been assigned, or none fit the request.",
            "parameters": {"type": "object", "properties": {}},
        }},
    ]
    return {"prompt": prompt, "functions": functions}


# ===========================================================================
# Tier 3 -- Tool-Agent
# ===========================================================================

def build_tool_agent_prompt(user_query: str, category: str, tool: str) -> dict[str, Any]:
    apis = d.get_apis(category, tool)
    if not apis:
        raise ValueError(f"Unknown (category, tool) pair: ('{category}', '{tool}').")
    ids = [a["endpoint_id"] for a in apis]

    prompt = f"""You are APIGPT with access to a database of APIs, categorized into
various sections. Each category contains multiple tools, and each tool
encompasses numerous APIs. Your task is to assist users in finding
relevant APIs within the tool "{tool}" of the "{category}" category. Use
the 'get_APIs_in_tool' function to list the APIs available, and the
'get_API_detail' function to see the full detail of specific APIs, before
deciding. You may also call 'check_if_request_solvable' to check whether
the APIs you have added so far are likely enough to solve the request.
Once you identify relevant API names, use the 'add_API_into_API_pool'
function to add them to the final API list. If you conclude that all
possible APIs have been explored, or if there are no relevant APIs in
this tool, invoke the 'finish_search' function. Keep your responses
concise, ideally within three sentences.

User request:
{user_query}"""

    functions = [
        {"type": "function", "function": {
            "name": "get_APIs_in_tool",
            "description": f"List the API endpoint_ids available in '{tool}'.",
            "parameters": {"type": "object", "properties": {
                "tool_name": {"type": "string", "enum": [tool]},
            }, "required": ["tool_name"]},
        }},
        {"type": "function", "function": {
            "name": "get_API_detail",
            "description": "Get full detail (method, path, fields) for specific endpoint_ids.",
            "parameters": {"type": "object", "properties": {
                "APIs": {"type": "array", "items": {"type": "string", "enum": ids}},
            }, "required": ["APIs"]},
        }},
        {"type": "function", "function": {
            "name": "check_if_request_solvable",
            "description": "Check whether the APIs added so far are likely enough to solve the request.",
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "add_API_into_API_pool",
            "description": "Add relevant endpoint_id(s) from this tool to the shared API candidate pool.",
            "parameters": {"type": "object", "properties": {
                "APIs": {"type": "array", "items": {"type": "string", "enum": ids}},
            }, "required": ["APIs"]},
        }},
        {"type": "function", "function": {
            "name": "finish_search",
            "description": "Call once all possible APIs have been explored, or none fit the request.",
            "parameters": {"type": "object", "properties": {}},
        }},
    ]
    return {"prompt": prompt, "functions": functions}


# ===========================================================================
# Self-Reflection re-activation messages
#
# Self-reflection does NOT rebuild a fresh prompt -- it re-activates an
# agent's OWN running conversation by appending one of these as a new
# turn, so the agent still remembers what it already explored instead of
# starting over from scratch. The pipeline appends the returned text as a
# new {"role": "user", ...} message onto that agent's own message list.
# ===========================================================================

def tool_reactivation_message(fail_reason: str) -> str:
    """Re-activating a Tool-Agent after the Solver couldn't use its pool."""
    return (
        f"The current APIs have failed to solve the query, resulting in: "
        f"{fail_reason}. You need to analyze this result and seek additional "
        f"APIs. It's possible that the tools lack the relevant APIs. In such "
        f"cases, you should call the 'finish_search' function. Remember not "
        f"to invent tool or API names."
    )


def category_reactivation_message(fail_reason: str) -> str:
    """Re-activating a Category-Agent after the Solver couldn't use its pool."""
    return (
        f"The current APIs have failed to solve the query, and the reason "
        f"is: {fail_reason}. Please consider assigning more unexplored tools "
        f"to the agents."
    )


def meta_reactivation_message(fail_reason: str) -> str:
    """Re-activating the Meta-Agent after the Solver couldn't use its pool."""
    return (
        f"The current APIs have failed to solve the query, and the reason "
        f"is: {fail_reason}. Please consider assigning more unexplored "
        f"categories to the agents."
    )


# ===========================================================================
# Solver
# ===========================================================================
SOLVER_INSTRUCTIONS = """You are AutoGPT, a precise API-calling assistant. A retrieval step has
already picked the small set of candidate APIs below for this specific
request. Read the user request once, then respond with exactly one JSON
object and nothing else (no markdown, no explanation).

Step 1 - pick the endpoint: choose the single best-matching candidate below
(path + HTTP method) for the user's request. Never invent or modify a
path, and never use an endpoint that is not listed among the candidates.
If none of the candidates can solve the request at all, respond with
{"error": "<short reason>"} instead of the shape below, and name the
"METHOD /path" of every candidate you are confident is irrelevant in an
"irrelevant_endpoints" array.

Step 2 - build the request_body.

GOLDEN RULE (ABSOLUTE): request_body may contain ONLY fields whose
values the user explicitly wrote in the request. The candidate's field
list tells you which field NAMES are allowed -- it does NOT mean they
should appear. Every field the user did not mention MUST be omitted.

- For every value the user gave, use that exact literal value
  (e.g. "Lab-Pool"), never a numeric ID and never a path/URL.
- NEVER output placeholders such as "<field>" or "<scope_type>".
- NEVER output null, empty strings, [], or {} as filler for fields
  the user did not mention.
- NEVER add tags, custom_fields, comments, tenant, owner, status,
  scope or any other field unless the user explicitly mentioned it.
- Ignore "required" vs "optional": if the user did not state a value,
  the field is not in request_body. Do not guess, default or complete.
- Use {} for GET and DELETE.

Before answering, check each key in request_body: can you point to
the exact words in the user request that provide its value? If not,
DELETE that key.

Example:
Request: "Create a new VLAN group named 'Lab-Pool' with slug 'lab-pool'
and description 'Isolated lab testing VLAN pool'."
Correct request_body:
{"name": "Lab-Pool", "slug": "lab-pool", "description": "Isolated lab testing VLAN pool"}
WRONG: adding scope_type, scope_id, tenant, owner, comments, tags,
custom_fields.

PATCH: include ONLY the fields the user explicitly wants to change.
Never include the record ID or unchanged fields in request_body.

GET requests -- query-string filters (ABSOLUTE):
- Every filter or condition the user mentions (e.g. "status is X",
  "named Y", "in site Z", "with role R") MUST be appended to "endpoint"
  as a query string, using the exact parameter name from the chosen
  candidate's "parameters" list: path + "?" + name=value, joined with "&".
- The word "all" does NOT mean "no filter". "Show me all prefixes,
  their status is container" still needs ?status=container.
- Use the bare path ONLY when the user gives no filter at all.
- URL-encode values (spaces -> %20, & -> %26). Use lowercase for enum
  values such as status (Container -> container).
- Only use parameter names that appear in the candidate's "parameters"
  list. Never guess defaults.
- request_body for GET is always {}.
- Examples:
    "Show me all prefixes, their status is container"
      -> "endpoint": "/api/ipam/prefixes/?status=container"
    "List devices in site London with status active"
      -> "endpoint": "/api/dcim/devices/?site=London&status=active"
    "Show all sites"
      -> "endpoint": "/api/dcim/sites/"

Record IDs (ABSOLUTE):
- If the user gives a concrete numeric ID anywhere in the request
  (e.g. "ID 56", "id: 56", "#56", "with ID 56"), you MUST replace
  "{id}" in the endpoint path with that exact number.
  Never leave "{id}" in the path when an ID was given.
  Never put the ID in request_body or "parameters".
- If the user gave NO numeric ID, keep "{id}" unchanged in the path.
  When they named the target by another attribute (name, slug, VID,
  address), add a top-level "parameters" object: {"<field>": "<value>"}.
- "parameters" is ONLY for locating a single record whose path contains
  "{id}" when no numeric ID was given. Never use it for list filters.
- Only numeric IDs count. A name like "Core-SW1" is NOT an ID.
- Examples:
    "Delete cable ID 56 connecting two terminations."
      -> {"title": "Delete cable", "method": "DELETE",
          "endpoint": "/api/dcim/cables/56/", "request_body": {}}
    "Update device 12: set status to active"
      -> {"title": "Update device", "method": "PATCH",
          "endpoint": "/api/dcim/devices/12/",
          "request_body": {"status": "active"}}
    "Delete the site named London"
      -> {"title": "Delete site", "method": "DELETE",
          "endpoint": "/api/dcim/sites/{id}/", "request_body": {},
          "parameters": {"name": "London"}}

Output shape (exactly this JSON object, nothing else):
{"title": "...", "method": "GET|POST|PUT|PATCH|DELETE", "endpoint": "...",
 "request_body": {...}, "parameters": {...} (optional)}

Candidate endpoints (choose only from these):
{candidates}
"""
_GIVE_UP_HINT = """

If you give up, you MAY include an "irrelevant_endpoints" array of the
exact "METHOD /path" strings (copied exactly as shown above) for every
candidate you are confident does NOT apply, e.g.:
{"error": "none of the candidates match this request", "irrelevant_endpoints": ["GET /api/ipam/vlans/"]}
Leave it out if you are unsure which candidates are irrelevant.
"""


def _candidate_blocks(candidate_keys: list[str]) -> list[str]:
    blocks = []
    for key in candidate_keys:
        method, _, endpoint = key.partition(" ")
        method, endpoint = method.strip().upper(), endpoint.strip()
        if not method or not endpoint:
            continue
        schema = find_endpoint_schema(d.raw_endpoints(), endpoint, method)
        if schema is None:
            continue  # candidate key doesn't match a real schema entry - skip rather than fabricate
        blocks.append(format_candidate_block(method, endpoint, schema))
    return blocks


def build_solver_prompt(user_query: str, candidate_endpoint_ids: list[str]) -> dict[str, Any]:
    if not candidate_endpoint_ids:
        raise ValueError("Empty candidate pool - nothing for the Solver to work with.")

    candidate_keys = d.keys_for_endpoint_ids(candidate_endpoint_ids)
    blocks = _candidate_blocks(candidate_keys)
    if not blocks:
        raise ValueError("None of the pooled endpoint_ids resolved to a real schema entry.")

    prompt = (
        SOLVER_INSTRUCTIONS.replace("{candidates}", "\n".join(blocks))
        + _GIVE_UP_HINT
        + f"\n\nUser request:\n{user_query}\n"
    )
    return {"prompt": prompt, "candidate_key_map": d.candidate_key_map_for_endpoint_ids(candidate_endpoint_ids)}


def build_solver_reactivation_prompt(candidate_endpoint_ids: list[str]) -> dict[str, Any]:
    """Self-Reflection in the Solver: re-activated with the UPDATED
    candidate pool, as a new turn appended to its own running conversation
    -- not a fresh prompt."""
    if not candidate_endpoint_ids:
        raise ValueError("Empty candidate pool after self-reflection.")

    candidate_keys = d.keys_for_endpoint_ids(candidate_endpoint_ids)
    blocks = _candidate_blocks(candidate_keys)
    if not blocks:
        raise ValueError("None of the pooled endpoint_ids resolved to a real schema entry.")

    message = (
        "Self-reflection: the previous candidate pool did not solve the "
        "request. Here is the updated candidate pool - choose again from "
        "these only, following the same output rules as before.\n\n"
        "Candidate endpoints (choose only from these):\n" + "\n".join(blocks)
        + _GIVE_UP_HINT
    )
    return {"message": message, "candidate_key_map": d.candidate_key_map_for_endpoint_ids(candidate_endpoint_ids)}
