
from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit


def get_case_text(case: dict) -> str:

    return (case.get("task") or case.get("question") or "").strip()


def normalize_path(path: str) -> str:
    """Makes two paths comparable regardless of a trailing slash."""
    path = path or ""
    return path if path.endswith("/") else path + "/"


def _split_path_and_query(path: str) -> tuple[str, dict[str, list[str]]]:

    split = urlsplit(path or "")
    query: dict[str, list[str]] = {}
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        query.setdefault(unquote(key), []).append(unquote(value))
    # sort each param's own values too, so repeated params ("a=1&a=2" vs
    # "a=2&a=1") also compare order-independently
    return split.path, {key: sorted(values) for key, values in query.items()}


def _paths_match(expected_path: str, actual_path: str) -> bool:
    """Two endpoints match when their base path is the same (trailing
    slash aside) and they carry the same query parameters -- same keys,
    same values -- regardless of the order either one lists them in."""
    expected_base, expected_query = _split_path_and_query(expected_path)
    actual_base, actual_query = _split_path_and_query(actual_path)
    return normalize_path(expected_base) == normalize_path(actual_base) and expected_query == actual_query


def _normalize_value(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _bodies_match(expected: dict | None, actual: dict | None) -> bool:
    expected, actual = expected or {}, actual or {}
    if set(expected) != set(actual):
        return False
    return all(_normalize_value(expected[key]) == _normalize_value(actual[key]) for key in expected)


def evaluate_case(expected_answer: dict, actual: dict | None) -> dict:
    """Compares one retrieved {method, endpoint, body} against one case's
    expected_answer. All three have to match for "correct"."""
    expected_method = (expected_answer.get("method") or "").strip().upper()
    expected_path = expected_answer.get("expected_endpoint") or ""
    expected_body = expected_answer.get("body")

    if actual is None:
        return {
            "correct": False,
            "method_match": False,
            "endpoint_match": False,
            "body_match": False,
            "reason": "pipeline did not return an answer",
        }

    actual_method = (actual.get("method") or "").strip().upper()
    actual_path = actual.get("endpoint") or ""
    actual_body = actual.get("body")

    method_match = expected_method == actual_method
    endpoint_match = _paths_match(expected_path, actual_path)
    # No body expected (GET/DELETE) -> actual must also have none. A body
    # is expected (POST/PATCH) -> actual's fields must match it exactly.
    body_match = (actual_body in (None, {})) if expected_body is None else _bodies_match(expected_body, actual_body)

    return {
        "correct": method_match and endpoint_match and body_match,
        "method_match": method_match,
        "endpoint_match": endpoint_match,
        "body_match": body_match,
    }
