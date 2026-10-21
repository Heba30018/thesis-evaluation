

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlsplit


def get_case_text(case: dict) -> str:
    """The ground-truth dataset uses "task" for write cases and "question"
    for read cases -- accept either."""
    return (case.get("task") or case.get("question") or "").strip()


def _normalize_endpoint(endpoint: str) -> tuple[str, dict]:
    if not endpoint:
        return "", {}
    parts = urlsplit(endpoint)
    path = parts.path if parts.path.endswith("/") else parts.path + "/"
    return path, parse_qs(parts.query, keep_blank_values=True)


def _endpoints_match(expected: str, actual: str) -> bool:
    return _normalize_endpoint(expected) == _normalize_endpoint(actual)


def _normalize_value(v: Any) -> Any:
    return v.strip() if isinstance(v, str) else v


def _bodies_match(expected: dict | None, actual: dict | None) -> bool:
    expected, actual = expected or {}, actual or {}
    if set(expected) != set(actual):
        return False
    return all(_normalize_value(expected[k]) == _normalize_value(actual[k]) for k in expected)


def evaluate_case(expected_answer: dict, actual: dict | None) -> dict:
    """`expected_answer` is ground_truth["expected_answer"][0]; `actual` is
    the Solver's parsed JSON answer (or None if the pipeline errored)."""
    expected_method = (expected_answer.get("method") or "").strip().upper()
    expected_endpoint = expected_answer.get("expected_endpoint") or ""
    expected_body = expected_answer.get("body")

    if actual is None:
        return {"correct": False, "endpoint_match": False, "method_match": False,
                "body_match": False, "reason": "pipeline did not return an answer"}

    actual_method = (actual.get("method") or "").strip().upper()
    actual_endpoint = actual.get("endpoint") or ""
    actual_body = actual.get("request_body")

    method_match = expected_method == actual_method
    endpoint_match = _endpoints_match(expected_endpoint, actual_endpoint)
    body_match = (actual_body in (None, {})) if expected_body is None else _bodies_match(expected_body, actual_body)

    return {
        "correct": method_match and endpoint_match and body_match,
        "endpoint_match": endpoint_match,
        "method_match": method_match,
        "body_match": body_match,
    }
