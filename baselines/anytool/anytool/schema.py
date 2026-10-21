"""Endpoint schema lookups used to build the Solver's prompt: given a
"METHOD /path" candidate, find its real requestBody fields in endpoints.json
so the Solver sees actual field names instead of guessing them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .data import DATA_DIR, merged_request_fields

ASSIGNED_OBJECT_FIELDS_FILE = DATA_DIR / "assigned_object_fields.json"

# Optional: candidate {type_key: endpoint} maps for polymorphic
# "assigned_object_type" / "assigned_object_id" field pairs (a handful of
# NetBox endpoints that can point at more than one kind of object). Missing
# file just means no endpoint gets this extra hint in its prompt block.
try:
    with ASSIGNED_OBJECT_FIELDS_FILE.open(encoding="utf-8") as f:
        ASSIGNED_OBJECT_FIELDS: dict[str, Any] = json.load(f)
except FileNotFoundError:
    ASSIGNED_OBJECT_FIELDS = {}


def find_endpoint_schema(endpoints: list[dict[str, Any]], endpoint: str, method: str) -> dict[str, Any] | None:
    method = method.strip().upper()
    for ep in endpoints:
        if str(ep.get("path", "")).strip() == endpoint and str(ep.get("method", "")).strip().upper() == method:
            return ep
    return None


def collection_path(endpoint: str) -> str:
    """Strips a trailing {id}-style segment so a detail path (.../{id}/)
    and its collection path (.../) both key into ASSIGNED_OBJECT_FIELDS the
    same way."""
    return re.sub(r"\{[^/]+\}.*", "", endpoint)


def format_candidate_block(method: str, endpoint: str, endpoint_schema: dict[str, Any]) -> str:
    """One candidate's entry in the Solver prompt: its fields, plus an
    assigned_object_fields hint when relevant."""
    fields = merged_request_fields(endpoint_schema)
    lines = [f"- {method} {endpoint}"]
    if fields:
        for name, value in fields.items():
            lines.append(f"    {name}: {value}")
    else:
        lines.append("    (no request-body fields)")

    hint = ASSIGNED_OBJECT_FIELDS.get(collection_path(endpoint))
    if hint:
        lines.append(f"    assigned_object_fields: {hint}")

    return "\n".join(lines)
