"""Loads endpoints.json and splits it into the three-tier hierarchy the
API retriever needs: category -> tool -> API.

Each endpoint in endpoints.json has an "endpoint_id" shaped like
"<category>.<tool>.<name>" (e.g. "circuits.circuit-types.list"). Splitting
on "." gives the category and tool directly, so no separate taxonomy file
is needed -- the dataset carries its own hierarchy.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ENDPOINTS_FILE = PROJECT_ROOT / "endpoints.json"

_lock = threading.RLock()  # _load() calls raw_endpoints() while already holding this lock
_entries: list[dict[str, Any]] | None = None
_tree: dict[str, dict[str, list[dict]]] | None = None
_by_id: dict[str, dict] | None = None


def raw_endpoints(endpoints_file: Path = ENDPOINTS_FILE) -> list[dict[str, Any]]:
    """The raw endpoints.json entries (one dict per API), exactly as
    written -- used by prompts.py/schema.py to look up an endpoint's real
    requestBody schema when building the Solver's candidate list."""
    global _entries
    if _entries is None:
        with _lock:
            if _entries is None:
                with endpoints_file.open(encoding="utf-8") as f:
                    raw = json.load(f)
                _entries = raw.get("endpoints", raw) if isinstance(raw, dict) else raw
    return _entries


def _load(endpoints_file: Path = ENDPOINTS_FILE) -> tuple[dict, dict]:
    """Splits raw_endpoints() into the retriever's two lookup shapes: a
    category -> tool -> [endpoint record] tree, and a flat endpoint_id ->
    record lookup. Thread-safe so several agents running at the same time
    can share one copy instead of each loading and parsing the file
    again."""
    global _tree, _by_id

    if _tree is not None:
        return _tree, _by_id  # type: ignore[return-value]

    with _lock:
        if _tree is not None:
            return _tree, _by_id  # type: ignore[return-value]

        tree: dict[str, dict[str, list[dict]]] = {}
        by_id: dict[str, dict] = {}

        for entry in raw_endpoints(endpoints_file):
            endpoint_id = str(entry.get("endpoint_id", "")).strip()
            parts = endpoint_id.split(".")
            category = parts[0] if parts and parts[0] else "unknown"
            tool = parts[1] if len(parts) > 1 else "unknown"

            record = {
                "endpoint_id": endpoint_id,
                "method": str(entry.get("method", "")).upper(),
                "path": str(entry.get("path", "")),
                "raw": entry,
            }
            tree.setdefault(category, {}).setdefault(tool, []).append(record)
            if endpoint_id:
                by_id[endpoint_id] = record

        _tree, _by_id = tree, by_id
        return _tree, _by_id


def merged_request_fields(endpoint_schema: dict[str, Any]) -> dict[str, Any]:
    """Required + optional request-body fields declared for one endpoint."""
    request_body = endpoint_schema.get("requestBody") or {}
    fields: dict[str, Any] = {}
    for section in ("required_fields", "optional_fields"):
        section_fields = request_body.get(section)
        if isinstance(section_fields, dict):
            fields.update(section_fields)
    return fields


def _field_summary(raw: dict[str, Any]) -> str:
    fields = merged_request_fields(raw)
    if fields:
        return ", ".join(list(fields.keys())[:6])
    params = (raw.get("parameters") or {}).get("optional_fields") or []
    names = [k for d in params for k in d if k != "suffix_set_id"]
    return ", ".join(names[:6]) if names else "(no fields)"


# ---------------------------------------------------------------------------
# Tier 1: categories (Meta-Agent)
# ---------------------------------------------------------------------------

def get_categories() -> list[dict]:
    tree, _ = _load()
    return [
        {"category": category, "tool_count": len(tools), "endpoint_count": sum(len(v) for v in tools.values())}
        for category, tools in sorted(tree.items())
    ]


# ---------------------------------------------------------------------------
# Tier 2: tools within a category (Category-Agent)
# ---------------------------------------------------------------------------

def get_tools_in_category(category: str) -> list[dict]:
    tree, _ = _load()
    tools = tree.get(category, {})
    return [{"tool": tool, "endpoint_count": len(entries)} for tool, entries in sorted(tools.items())]


def get_tool_descriptions(category: str, tool_list: list[str]) -> list[dict]:
    """Short description (endpoint count, HTTP methods, common fields) for
    one or more tools -- used by both the Meta-Agent and the Category-Agent
    to judge a tool without having to call back and forth a second time."""
    tree, _ = _load()
    tools = tree.get(category, {})
    out = []
    for tool in tool_list:
        entries = tools.get(tool, [])
        methods = sorted({e["method"] for e in entries})
        fields: list[str] = []
        for e in entries:
            for f in _field_summary(e["raw"]).split(", "):
                if f and f not in fields:
                    fields.append(f)
        out.append({
            "tool": tool,
            "endpoint_count": len(entries),
            "methods": methods,
            "common_fields": ", ".join(fields[:8]) if fields else "(no fields)",
        })
    return out


# ---------------------------------------------------------------------------
# Tier 3: APIs within a tool (Tool-Agent)
# ---------------------------------------------------------------------------

def get_apis(category: str, tool: str) -> list[dict]:
    tree, _ = _load()
    entries = tree.get(category, {}).get(tool, [])
    return [
        {"endpoint_id": e["endpoint_id"], "method": e["method"], "path": e["path"], "fields_summary": _field_summary(e["raw"])}
        for e in entries
    ]


def get_APIs_in_tool(category: str, tool: str) -> list[dict]:
    """Bare id/method/path listing -- the Tool-Agent's lightweight first
    look at what's inside its tool(s), before calling 'get_API_detail' for
    the ones worth a closer look."""
    tree, _ = _load()
    entries = tree.get(category, {}).get(tool, [])
    return [{"endpoint_id": e["endpoint_id"], "method": e["method"], "path": e["path"]} for e in entries]


def get_API_detail(endpoint_ids: list[str]) -> list[dict]:
    """Full detail for specific endpoint_ids, possibly spanning more than
    one tool (a single Tool-Agent can be responsible for several tools at
    once)."""
    _, by_id = _load()
    out = []
    for eid in endpoint_ids:
        record = by_id.get(str(eid).strip())
        if record is None:
            continue
        out.append({
            "endpoint_id": record["endpoint_id"],
            "method": record["method"],
            "path": record["path"],
            "fields_summary": _field_summary(record["raw"]),
        })
    return out


# ---------------------------------------------------------------------------
# Resolving the candidate pool (collected endpoint_ids) for the Solver
# ---------------------------------------------------------------------------

def get_api_record(endpoint_id: str) -> dict | None:
    _, by_id = _load()
    return by_id.get(str(endpoint_id).strip())


def keys_for_endpoint_ids(endpoint_ids: list[str]) -> list[str]:
    """endpoint_id list -> ["METHOD /path", ...], the form the Solver's
    prompt is built from. Unknown ids are silently skipped."""
    _, by_id = _load()
    keys = []
    for eid in endpoint_ids:
        record = by_id.get(str(eid).strip())
        if record is not None:
            keys.append(f"{record['method']} {record['path']}")
    return keys


def candidate_key_map_for_endpoint_ids(endpoint_ids: list[str]) -> dict[str, str]:
    """endpoint_id -> "METHOD /path", so a Solver "give up" naming an
    irrelevant candidate (by the only form its prompt shows it, "METHOD
    /path") can be translated back to the endpoint_id to prune."""
    _, by_id = _load()
    out = {}
    for eid in endpoint_ids:
        record = by_id.get(str(eid).strip())
        if record is not None:
            out[record["endpoint_id"]] = f"{record['method']} {record['path']}"
    return out
