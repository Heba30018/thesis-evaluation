import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit, parse_qs


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_ground_truth(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of ground-truth cases")
    return data


def parse_protocol_message(obj):
    """Given one parsed JSON object from the bridge's stdout (one line of the
    --netbox-ui-bridge JSONL protocol), return a normalized actual-result
    entry {user_request, method, endpoint, body} if it's a usable action,
    or {"user_request":..., "needs_disambiguation": True} if it's an
    unresolved disambiguation card, or None if the line isn't an action at
    all (a "chunk"/"done"/"error" progress line)."""
    if not isinstance(obj, dict) or obj.get("type") != "json_request":
        return None
    message = obj.get("message")
    if not isinstance(message, dict):
        return None

    action = message.get("action")

    if action == "netbox_request":
        # Built by build_reviewable_netbox_action(): a fully-resolved
        # POST/PUT/PATCH/DELETE request, ready for review/execution.
        return {
            "user_request": message.get("user_prompt"),
            "method": message.get("method"),
            "endpoint": message.get("endpoint"),
            "body": message.get("request_body"),
        }

    if action == "netbox_result":
        # Built by build_get_result_action(): a fully-resolved GET request
        # whose result has already been fetched. No request body to compare.
        return {
            "user_request": message.get("user_prompt"),
            "method": message.get("method", "GET"),
            "endpoint": message.get("endpoint"),
            "body": None,
        }

    if action == "disambiguation":
        # Built by build_disambiguation_action(): the agent found several
        # matching candidates and is waiting for the user to pick one. Not a
        # final answer yet, so it can't be auto-graded against ground truth.
        return {"user_request": message.get("user_prompt"), "needs_disambiguation": True}

    return None


def load_actual_results(path: Path):
    """Accepts, auto-detected:
      (a) a clean JSON array/object of pre-built actual-result entries, or
      (b) the raw stdout of `smartclai --netbox-ui-bridge` (one JSON object
          per line; every "json_request" line whose message.action is
          "netbox_request" or "netbox_result" is extracted), or
      (c) [legacy] a raw log containing '[CARD]\\n{...}' blocks, from the
          old print_eval_card() debug helper (kept only for backward
          compatibility with older runs/binaries)."""
    text = path.read_text(encoding="utf-8")

    # (a) Clean JSON first.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # (b) Line-delimited bridge protocol: one JSON object per line.
    entries = []
    disambiguation_count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed = parse_protocol_message(obj)
        if parsed is None:
            continue
        if parsed.get("needs_disambiguation"):
            disambiguation_count += 1
            continue
        entries.append(parsed)

    if entries:
        if disambiguation_count:
            print(f"Note: {disambiguation_count} case(s) hit a disambiguation "
                  f"prompt (multiple matching candidates) and could not be "
                  f"auto-evaluated - they need a candidate picked first.")
        return entries

    # (c) Legacy fallback: pull out every "[CARD]\n{...}" block.
    decoder = json.JSONDecoder()
    marker = "[CARD]"
    idx = 0
    while True:
        pos = text.find(marker, idx)
        if pos == -1:
            break
        brace_start = text.find("{", pos)
        if brace_start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace_start)
            entries.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx = pos + len(marker)

    if not entries:
        raise ValueError(
            f"{path}: could not parse as a JSON array/object, found no "
            f"bridge protocol 'json_request' lines, and found no legacy "
            f"'[CARD]' blocks. Check the file format."
        )
    return entries


# --------------------------------------------------------------------------
# Matching actual results back to ground-truth cases
# --------------------------------------------------------------------------
# NOTE: we deliberately do NOT match on "id". print_eval_card() assigns a
# random 4-digit id (generate_eval_case_id()) that has no relationship to
# the ground-truth "CRUD-xxx"/"GT-xxx" id, so an id match would be
# meaningless. Instead: run each ground-truth case's "task" text as the
# query, and match the actual result back to it by that exact text
# ("user_request" in the actual entry == "task" in the ground truth).

def build_actual_index(actual_entries):
    by_request_text = {}
    for entry in actual_entries:
        req_text = entry.get("user_request") or entry.get("task")
        if req_text:
            by_request_text[req_text.strip()] = entry
    return by_request_text


def get_case_text(case):
    """The ground-truth dataset is inconsistent: GT-xxx (GET) cases use the
    key "question", CRUD-xxx (POST/PATCH/DELETE) cases use "task". Accept
    either."""
    return (case.get("task") or case.get("question") or "").strip()


def find_actual_for_case(case, by_request_text):
    task_text = get_case_text(case)
    if task_text and task_text in by_request_text:
        return by_request_text[task_text]
    return None


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def get_first(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_endpoint(endpoint):
    """Split an endpoint into (path, query-dict) so that query-parameter
    order never affects comparison."""
    if not endpoint:
        return "", {}
    parts = urlsplit(endpoint)
    path = parts.path
    if not path.endswith("/"):
        path += "/"
    query = parse_qs(parts.query, keep_blank_values=True)
    return path, query


def endpoints_match(expected_endpoint, actual_endpoint):
    exp_path, exp_query = normalize_endpoint(expected_endpoint)
    act_path, act_query = normalize_endpoint(actual_endpoint)
    return exp_path == act_path and exp_query == act_query


def normalize_value(v):
    if isinstance(v, str):
        return v.strip()
    return v


def bodies_match(expected_body, actual_body):
    """Order-independent, field-by-field dict comparison. Returns
    (is_match, list_of_human_readable_mismatches)."""
    expected_body = expected_body or {}
    actual_body = actual_body or {}
    mismatches = []

    expected_keys = set(expected_body.keys())
    actual_keys = set(actual_body.keys())

    for key in sorted(expected_keys - actual_keys):
        mismatches.append(f"missing field '{key}' (expected {expected_body[key]!r})")
    for key in sorted(actual_keys - expected_keys):
        mismatches.append(f"unexpected extra field '{key}' (got {actual_body[key]!r})")
    for key in sorted(expected_keys & actual_keys):
        if normalize_value(expected_body[key]) != normalize_value(actual_body[key]):
            mismatches.append(
                f"field '{key}': expected {expected_body[key]!r}, got {actual_body[key]!r}"
            )

    return (len(mismatches) == 0), mismatches


def evaluate_case(expected_answer_entry, actual_entry):
    expected_method = (expected_answer_entry.get("method") or "").strip().upper()
    expected_endpoint = expected_answer_entry.get("expected_endpoint") or ""
    expected_body = expected_answer_entry.get("body")

    actual_method = (get_first(actual_entry, "expected_method", "method") or "").strip().upper()
    actual_endpoint = get_first(actual_entry, "expected_endpoint", "endpoint") or ""
    actual_body = get_first(actual_entry, "expected_request_body", "body", "request_body")

    method_match = expected_method == actual_method
    endpoint_match = endpoints_match(expected_endpoint, actual_endpoint)

    if expected_body is None:
        # GET/DELETE cases: no body expected at all.
        body_match = actual_body in (None, {})
    else:
        body_match, _mismatches = bodies_match(expected_body, actual_body)

    correct = method_match and endpoint_match and body_match

    return {
        "correct": correct,
        "endpoint_match": endpoint_match,
        "method_match": method_match,
        "body_match": body_match,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", default="test.json", type=Path,
                         help="Path to the ground-truth dataset (default: test.json)")
    parser.add_argument("--actual", required=True, type=Path,
                         help="Path to the actual pipeline output: a JSON array, or a raw "
                              "log file containing [CARD] blocks")
    parser.add_argument("--output", type=Path, default=None,
                         help="Where to write the updated ground-truth file. "
                              "Defaults to overwriting --ground-truth in place.")
    args = parser.parse_args()

    output_path = args.output or args.ground_truth

    ground_truth = load_ground_truth(args.ground_truth)
    actual_entries = load_actual_results(args.actual)
    by_request_text = build_actual_index(actual_entries)

    print(f"Loaded {len(ground_truth)} ground-truth cases from {args.ground_truth}")
    print(f"Loaded {len(actual_entries)} actual result(s) from {args.actual}")

    ran_count = 0
    correct_count = 0
    unmatched_ids = []

    for case in ground_truth:
        actual_entry = find_actual_for_case(case, by_request_text)

        if actual_entry is None:
            case["evaluation"] = {"correct": None, "reason": "no matching actual result found (task text didn't match any user_request)"}
            unmatched_ids.append(case.get("id"))
            continue

        expected_answers = case.get("expected_answer") or []
        if not expected_answers:
            case["evaluation"] = {"correct": None, "reason": "ground truth has no expected_answer"}
            continue

        # Most cases have exactly one expected call; if there are several,
        # this evaluates against the first one (extend here if you add
        # multi-step ground truth later).
        result = evaluate_case(expected_answers[0], actual_entry)
        case["evaluation"] = result

        ran_count += 1
        if result["correct"]:
            correct_count += 1

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Evaluated: {ran_count} / {len(ground_truth)} cases had a matching actual result")
    if ran_count:
        print(f"Correct:   {correct_count} / {ran_count}  ({100 * correct_count / ran_count:.1f}%)")
    if unmatched_ids:
        print(f"No actual result found for {len(unmatched_ids)} case(s): "
              f"{', '.join(str(i) for i in unmatched_ids[:20])}"
              f"{' ...' if len(unmatched_ids) > 20 else ''}")
    print(f"Results written to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
