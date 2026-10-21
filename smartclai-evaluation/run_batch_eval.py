#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_agents as ea  # noqa: E402  (reuse its matching/comparison logic)

try:
    import tiktoken
    _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TOKEN_ENCODING = None


def count_tokens(text: str) -> int:
    """Token count for one piece of prompt/completion text, via tiktoken's
    cl100k_base encoding. The bridge (netbox_ui_wrapper.cpp's emit_metrics())
    gives us the raw prompt/output strings for Agent 1 and Agent 2, not an
    OpenAI "usage" object - app.bin's stdio streaming protocol never surfaces
    one - so token counts are computed locally instead."""
    if not text:
        return 0
    if _TOKEN_ENCODING is None:
        sys.exit(
            "tiktoken is required to count Agent 1 / Agent 2 tokens.\n"
            "Install it with:  pip install tiktoken"
        )
    return len(_TOKEN_ENCODING.encode(text))


def drain_stderr(pipe, buffer: deque):
    """Keep the bridge's stderr pipe from filling up (which would make the
    bridge block trying to write to it) by continuously reading it into a
    small ring buffer we can print if something goes wrong. Nothing is
    written to disk."""
    try:
        for line in pipe:
            buffer.append(line.rstrip("\n"))
    except Exception:
        pass


def read_case_response(proc, request_id, timeout_lines=100000):
    """Read the bridge's stdout, line by line, for this request_id.
    Returns (actual_entry_or_None, status_string, metrics_or_None,
    resolution_metrics_or_None).

    metrics is the "metrics" line emitted by emit_metrics() right after
    Agent 2 returns (Agent 1 / Agent 2 prompt+output text and per-agent
    time_s). resolution_metrics is the "resolution_metrics" line emitted
    once every planner action has finished running through the
    deterministic resolution/NetBox-call layer (resolve_path_candidates(),
    resolve_filter_ids_in_endpoint(), callNetbox(), ...) - i.e. after the
    actionable json_request line(s) for this case. So once we have an
    actionable entry we keep reading (without overwriting it) until
    resolution_metrics, "done", or "error" closes out this request, instead
    of returning immediately."""
    metrics = None
    resolution_metrics = None
    actionable = None

    for _ in range(timeout_lines):
        line = proc.stdout.readline()
        if line == "":
            if actionable is not None:
                return actionable, "ok", metrics, resolution_metrics
            return None, "bridge closed stdout unexpectedly", metrics, resolution_metrics
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("id") not in (request_id, None):
            # Response for a different request id - shouldn't normally
            # happen since the bridge is synchronous, but skip defensively
            # rather than mis-attributing it.
            continue

        obj_type = obj.get("type")

        if obj_type == "metrics":
            metrics = obj
            continue

        if obj_type == "resolution_metrics":
            resolution_metrics = obj
            if actionable is not None:
                return actionable, "ok", metrics, resolution_metrics
            continue

        if obj_type == "error":
            if actionable is not None:
                return actionable, "ok", metrics, resolution_metrics
            return None, obj.get("message", "unknown error"), metrics, resolution_metrics

        if obj_type == "done":
            if actionable is not None:
                return actionable, "ok", metrics, resolution_metrics
            # "done" with nothing actionable seen yet: keep waiting (matches
            # the original behaviour - the bridge is synchronous, so this
            # shouldn't normally happen before an actionable line or error).
            continue

        parsed = ea.parse_protocol_message(obj)
        if parsed is None:
            continue  # progress noise (chunk / agent_title / etc.)
        if parsed.get("needs_disambiguation"):
            return None, "needs disambiguation (multiple matching candidates)", metrics, resolution_metrics
        if actionable is None:
            actionable = parsed
        # Keep reading - resolution_metrics / "done" for this id still to come.

    if actionable is not None:
        return actionable, "ok", metrics, resolution_metrics
    return None, "gave up waiting for a response (too many lines)", metrics, resolution_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default="../build/smartclai", type=Path,
                         help="Path to the smartclai executable (default: ../build/smartclai, "
                              "since this script lives in evaluation/, one level below "
                              "smartclai-client/)")
    parser.add_argument("--ground-truth", default="test.json", type=Path,
                         help="Path to test.json (default: test.json)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only send the first N tasks. Use this for a quick dry run before "
                              "committing to all 500.")
    args = parser.parse_args()

    if not args.binary.exists():
        sys.exit(f"Binary not found: {args.binary}  (build it first, or pass --binary <path>)")
    if not args.ground_truth.exists():
        sys.exit(f"Ground-truth file not found: {args.ground_truth}")

    with args.ground_truth.open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    cases = ground_truth[: args.limit] if args.limit else ground_truth

    print(f"Loaded {len(ground_truth)} case(s) from {args.ground_truth}"
          + (f" (running the first {len(cases)})" if args.limit else ""))
    print(f"Launching: {args.binary} --netbox-ui-bridge")
    print()

    # The bridge only emits "metrics"/"resolution_metrics" lines when this
    # variable is set (they are hidden from the Electron UI otherwise).
    bridge_env = dict(os.environ, SMARTCLAI_EMIT_METRICS="1")

    proc = subprocess.Popen(
        [str(args.binary), "--netbox-ui-bridge"],
        env=bridge_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    stderr_buffer = deque(maxlen=50)
    stderr_thread = threading.Thread(target=drain_stderr, args=(proc.stderr, stderr_buffer), daemon=True)
    stderr_thread.start()

    ran = 0
    correct = 0
    incorrect = 0
    metrics_rows = []

    try:
        for i, case in enumerate(cases, start=1):
            case_id = case.get("id", str(i))
            task_text = ea.get_case_text(case)
            label = f"[{i}/{len(cases)}] {case_id}"

            if not task_text:
                print(f"{label}: SKIPPED (no 'task' or 'question' field)")
                continue

            expected_answers = case.get("expected_answer") or []
            if not expected_answers:
                case["evaluation"] = {"correct": None, "reason": "ground truth has no expected_answer"}
                print(f"{label}: SKIPPED (no expected_answer in ground truth)")
                continue

            request = {"type": "ask", "id": str(case_id), "prompt": task_text}
            try:
                proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except BrokenPipeError:
                print(f"\n{label}: the bridge process is no longer running. Stopping.")
                break

            actual_entry, status, metrics, resolution_metrics = read_case_response(proc, str(case_id))

            if metrics is not None:
                agent1_prompt_tokens = count_tokens(metrics.get("agent1_prompt", ""))
                agent1_completion_tokens = count_tokens(metrics.get("agent1_output", ""))
                agent2_prompt_tokens = count_tokens(metrics.get("agent2_prompt", ""))
                agent2_completion_tokens = count_tokens(metrics.get("agent2_output", ""))
                agent1_time_s = metrics.get("agent1_time_s", 0.0)
                agent2_time_s = metrics.get("agent2_time_s", 0.0)
                # 0.0 if resolution_metrics never arrived (e.g. the planner
                # returned zero actions, or this case errored out before the
                # resolution stage ran) - older bridge binaries before this
                # feature won't send it either.
                resolution_time_s = (
                    resolution_metrics.get("resolution_time_s", 0.0)
                    if resolution_metrics is not None else 0.0
                )
                metrics_rows.append({
                    "id": case_id,
                    "agent1_time_s": agent1_time_s,
                    "agent2_time_s": agent2_time_s,
                    "resolution_time_s": resolution_time_s,
                    "total_time_s": agent1_time_s + agent2_time_s + resolution_time_s,
                    "agent1_prompt_tokens": agent1_prompt_tokens,
                    "agent1_completion_tokens": agent1_completion_tokens,
                    "agent2_prompt_tokens": agent2_prompt_tokens,
                    "agent2_completion_tokens": agent2_completion_tokens,
                })

            if actual_entry is None:
                case["evaluation"] = {"correct": None, "reason": status}
                print(f"{label}: NOT EVALUATED ({status})")
            else:
                result = ea.evaluate_case(expected_answers[0], actual_entry)
                case["evaluation"] = result
                ran += 1
                if result["correct"]:
                    correct += 1
                    print(f"{label}: correct")
                else:
                    incorrect += 1
                    print(f"{label}: INCORRECT "
                          f"(endpoint={result['endpoint_match']}, "
                          f"method={result['method_match']}, "
                          f"body={result['body_match']})")

            # Save after every single case, so an interruption never loses
            # progress already made.
            with args.ground_truth.open("w", encoding="utf-8") as f:
                json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    except KeyboardInterrupt:
        print("\nInterrupted - results saved for everything evaluated so far.")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=5) if proc.poll() is None else None

    if proc.returncode not in (0, None) and stderr_buffer:
        print("\nThe bridge process reported an error. Last lines of its stderr:")
        print("-" * 60)
        print("\n".join(stderr_buffer))
        print("-" * 60)

    print()
    print("=" * 60)
    print(f"Evaluated: {ran} / {len(cases)}")
    if ran:
        print(f"Correct:   {correct} / {ran}  ({100 * correct / ran:.1f}%)")
        print(f"Incorrect:   {incorrect} / {ran}  ({100 * incorrect / ran:.1f}%)")

    print(f"Results written to: {args.ground_truth}")
    print("=" * 60)

    if metrics_rows:
        n = len(metrics_rows)

        def avg(key):
            return sum(row[key] for row in metrics_rows) / n

        agent1_prompt_avg = avg("agent1_prompt_tokens")
        agent1_completion_avg = avg("agent1_completion_tokens")
        agent1_time_avg = avg("agent1_time_s")

        agent2_prompt_avg = avg("agent2_prompt_tokens")
        agent2_completion_avg = avg("agent2_completion_tokens")
        agent2_time_avg = avg("agent2_time_s")

        resolution_time_avg = avg("resolution_time_s")

        total_prompt_avg = agent1_prompt_avg + agent2_prompt_avg
        total_completion_avg = agent1_completion_avg + agent2_completion_avg
        total_time_avg = avg("total_time_s")  # Agent 1 + Agent 2 + Resolution

        print()
        print("=" * 60)
        print(f"Token / latency metrics (averaged over {n} case(s) with metrics)")
        print("-" * 60)
        print("Agent 1 (endpoint selector):")
        print(f"  Avg prompt tokens:      {agent1_prompt_avg:.2f}")
        print(f"  Avg completion tokens:  {agent1_completion_avg:.2f}")
        print(f"  Avg time:               {agent1_time_avg:.3f}s")
        print()
        print("Agent 2 (planner):")
        print(f"  Avg prompt tokens:      {agent2_prompt_avg:.2f}")
        print(f"  Avg completion tokens:  {agent2_completion_avg:.2f}")
        print(f"  Avg time:               {agent2_time_avg:.3f}s")
        print()
        print("Resolution (deterministic resolve/NetBox-call layer, no LLM):")
        print(f"  Avg time:               {resolution_time_avg:.3f}s")
        print()
        print("Total (Agent 1 + Agent 2 + Resolution):")
        print(f"  Avg prompt tokens:      {total_prompt_avg:.2f}")
        print(f"  Avg completion tokens:  {total_completion_avg:.2f}")
        print(f"  Avg total tokens:       {total_prompt_avg + total_completion_avg:.2f}")
        print(f"  Avg total time:         {total_time_avg:.3f}s")
        print("=" * 60)
    else:
        print("\nNo per-agent metrics were recorded (older bridge binary?). "
              "Rebuild SmartCLAI to pick up emit_metrics().")


if __name__ == "__main__":
    main()
