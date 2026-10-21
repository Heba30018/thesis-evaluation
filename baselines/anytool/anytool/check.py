"""Deterministic check: after the Solver proposes a (method, endpoint),
verify it against endpoints.json instead of trusting the model. A numeric
id in the path (e.g. "/api/.../7/") is canonicalized back to "{id}" first,
since the schema is keyed by the templated path.
"""

from __future__ import annotations

import re

from .data import raw_endpoints
from .schema import find_endpoint_schema

_ID_SEGMENT = re.compile(r"/\d+(?=/|$)")


def canonicalize(endpoint: str) -> str:
    return _ID_SEGMENT.sub("/{id}", endpoint.split("?", 1)[0])


def check_solver_output(method: str, endpoint: str) -> dict:
    canon = canonicalize(endpoint)
    schema = find_endpoint_schema(raw_endpoints(), canon, method.strip().upper())
    return {"valid": schema is not None, "canonical_endpoint": canon}
