#!/usr/bin/env python3
"""
Usage:
    export OPENAI_API_KEY=sk-...
    python evaluate.py --limit 20            # quick dry run
    python evaluate.py                       # all 500 cases
    python evaluate.py --limit 20 --output results/run1.json
    python evaluate.py --limit 20 --log-file results/run1_trace.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
log_file = Path(__file__).resolve().parent / "pipeline.log"
_log_fh = open(log_file, "a", encoding="utf-8")


def log(message: str = "") -> None:
    """print(), plus append the same line to pipeline.log."""
    print(message)
    _log_fh.write(message + "\n")
    _log_fh.flush()

load_dotenv()

from anytool.evaluation import evaluate_case, get_case_text  # noqa: E402
from anytool.pipeline import run_anytool  # noqa: E402

DEFAULT_GROUND_TRUTH = Path(__file__).resolve().parent / "data" / "ground_truth.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "evaluation.json"

# gpt-4o-mini's own per-call token price (https://developers.openai.com/docs/models/gpt-4o-mini),
# same figures used by the ReInvoke baseline's query.py/evaluate.py --
# used only to turn a run's token usage into a $ estimate.
PRICE_PER_1M_INPUT_TOKENS = 0.15
PRICE_PER_1M_OUTPUT_TOKENS = 0.60


AGENT_ROLES = ("meta", "category", "tool", "solver")


def _empty_agent_stats() -> dict:
    return {role: {"seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0} for role in AGENT_ROLES}


def _sum_agent_stats(agent_stats: dict) -> dict:

    return {
        "seconds": sum(s["seconds"] for s in agent_stats.values()),
        "prompt_tokens": sum(s["prompt_tokens"] for s in agent_stats.values()),
        "completion_tokens": sum(s["completion_tokens"] for s in agent_stats.values()),
    }


def _format_agent_stats_block(indent: str, agent_stats: dict, total: dict) -> str:
    """Multi-line "role: Ns  P in / C out" block, one line per role plus a
    closing total line -- used both per-case (exact integer token counts)
    and for the final, averaged summary (token counts are then an average
    across all queries, so a float -- :.0f rounds either one the same
    way)."""
    lines = [
        f"{indent}{role + ':':<10} {stats['seconds']:5.2f}s   {stats['prompt_tokens']:5.0f} in / {stats['completion_tokens']:4.0f} out"
        for role, stats in agent_stats.items()
    ]
    lines.append(
        f"{indent}{'total:':<10} {total['seconds']:5.2f}s   {total['prompt_tokens']:5.0f} in / {total['completion_tokens']:4.0f} out"
    )
    return "\n".join(lines)


class _Tee:
    """A write-only file-like object that forwards everything written to it
    to several real streams at once. Swapping this in for sys.stdout is how
    every plain print() in the codebase ends up both on the terminal and in
    the log file, without changing any of those print() calls themselves."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-file", type=Path, default=None,
                         help="Where to also save the full trace (default: --output with a .log extension).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    log_file_path = args.log_file or args.output.with_suffix(".log")
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    real_stdout = sys.stdout
    with log_file_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = _Tee(real_stdout, log_file)
        try:
            ran, correct = _run(args, log_file_path)
        finally:
            sys.stdout = real_stdout

    return 0


# def _run(args: argparse.Namespace, log_file_path: Path) -> tuple[int, int]:
#     print(f"(full trace is also being written to: {log_file_path})\n")

#     cases = json.loads(args.ground_truth.read_text(encoding="utf-8"))
#     if args.limit:
#         cases = cases[: args.limit]

#     print(f"Loaded {len(cases)} case(s) from {args.ground_truth}\n")

#     ran = correct = 0
#     started = time.time()

#     # Summed across every case that actually ran run_anytool(), so the end
#     # summary can report per-query averages.
#     total_cost = 0.0

#     # Same idea, but broken down by which part of the pipeline spent the
#     # time and tokens -- one running total per role (meta/category/tool/
#     # solver), each itself a {"seconds", "prompt_tokens",
#     # "completion_tokens"} bundle. See anytool/pipeline.py's
#     # RunState.agent_stats for where this is actually measured.
#     total_agent_stats = _empty_agent_stats()

#     for i, case in enumerate(cases, start=1):
#         case_id = case.get("id", str(i))
#         task_text = get_case_text(case)
#         label = f"[{i}/{len(cases)}] {case_id}"

#         expected_answers = case.get("expected_answer") or []
#         if not task_text or not expected_answers:
#             case["evaluation"] = {"correct": None, "reason": "missing task text or expected_answer"}
#             print(f"{label}: skipped")
#             continue

#         print(f"\n----- {label} ----- {task_text!r}")

#         try:
#             result = run_anytool(task_text, model=args.model)
#         except Exception as exc:
#             result = None
#         agent_stats = result.agent_stats
#         agent_total = _sum_agent_stats(agent_stats)

#         case_latency = agent_total["seconds"]
#         case_prompt_tokens = agent_total["prompt_tokens"]
#         case_completion_tokens = agent_total["completion_tokens"]
#         case_cost = (
#             case_prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
#             + case_completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT_TOKENS
#         )
#         total_cost += case_cost
#         for role in AGENT_ROLES:
#             total_agent_stats[role]["seconds"] += agent_stats[role]["seconds"]
#             total_agent_stats[role]["prompt_tokens"] += agent_stats[role]["prompt_tokens"]
#             total_agent_stats[role]["completion_tokens"] += agent_stats[role]["completion_tokens"]

#         evaluation = evaluate_case(expected_answers[0], result.answer if result.ok else None)
#         if not result.ok:
#             evaluation["reason"] = result.error
#         evaluation["latency_seconds"] = round(case_latency, 3)
#         evaluation["prompt_tokens"] = case_prompt_tokens
#         evaluation["completion_tokens"] = case_completion_tokens
#         evaluation["cost"] = case_cost
#         evaluation["agent_stats"] = agent_stats  # per-role breakdown: {"meta": {...}, "category": {...}, ...}
#         case["evaluation"] = evaluation
#         case["anytool_rounds_used"] = result.rounds_used

#         ran += 1
#         if evaluation["correct"]:
#             correct += 1

#         print(
#             f"----- {label}: {'correct' if evaluation['correct'] else 'incorrect'}  "
#             f"[{case_latency:.2f}s, ${case_cost:.6f}] -----"
#         )
#         print("      agent stats:")
#         print(_format_agent_stats_block("        ", agent_stats, agent_total))

#         # Save after every case so an interruption never loses progress.
#         args.output.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

#     elapsed = time.time() - started

#     # The run's grand total time/tokens is just every role's own running
#     # total (total_agent_stats), summed together -- every LLM call made
#     # during the whole run falls under exactly one role, so nothing is
#     # left out and nothing is double-counted.
#     total_agent_total = _sum_agent_stats(total_agent_stats)
#     avg_latency = (total_agent_total["seconds"] / ran) if ran else 0.0
#     avg_prompt_tokens = (total_agent_total["prompt_tokens"] / ran) if ran else 0.0
#     avg_completion_tokens = (total_agent_total["completion_tokens"] / ran) if ran else 0.0
#     avg_cost = total_cost / ran if ran else 0.0

#     # Same per-role breakdown as total_agent_stats, now divided by `ran` --
#     # the average, per query, that each role (and the grand total) cost.
#     avg_agent_stats = {
#         role: {
#             "seconds": (total_agent_stats[role]["seconds"] / ran) if ran else 0.0,
#             "prompt_tokens": (total_agent_stats[role]["prompt_tokens"] / ran) if ran else 0.0,
#             "completion_tokens": (total_agent_stats[role]["completion_tokens"] / ran) if ran else 0.0,
#         }
#         for role in AGENT_ROLES
#     }

#     print("\n" + "=" * 60)
#     print(f"Evaluated: {ran} / {len(cases)}  in {elapsed:.0f}s")
#     if ran:
#         print(f"Correct: {correct}/{ran}  ({100 * correct / ran:.1f}%)")
#     # print(f"\nToken usage & latency:")
#     # print(f"  Average Input Tokens/1Q:  {avg_prompt_tokens:.0f}")
#     # print(f"  Average Output Tokens/1Q: {avg_completion_tokens:.0f}")
#     # print(f"  Average Latency/1Q:       {avg_latency:.2f}s")
#     # print(f"  Average Cost/1Q:          ${avg_cost:.6f}")
#     # print(f"  Total Cost ({ran}Q):       ${total_cost:.6f}")
#     print(f"\nAgent stats (total across {ran} quer{'y' if ran == 1 else 'ies'}):")
#     print(_format_agent_stats_block("  ", total_agent_stats, total_agent_total))


#     print("=" * 60)

#     return ran, correct


def _run(args: argparse.Namespace, log_file_path: Path) -> tuple[int, int]:
    print(f"(full trace is also being written to: {log_file_path})\n")

    cases = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    print(f"Loaded {len(cases)} case(s) from {args.ground_truth}\n")

    # Load whatever is already saved in the output file (results from
    # earlier runs) instead of starting from scratch. New results get
    # appended onto this list, and every save below writes the *whole*
    # growing list back out -- so earlier runs' entries are never lost or
    # overwritten, only added to.
    if args.output.exists():
        results = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        results = []

    already_done_ids = {c.get("id") for c in results if c.get("id") is not None}
    cases_to_run = [c for c in cases if c.get("id") not in already_done_ids]
    skipped_already_done = len(cases) - len(cases_to_run)

    if skipped_already_done:
        print(
            f"{skipped_already_done} case(s) already present in {args.output} "
            f"-- skipping them, only running the remaining {len(cases_to_run)}.\n"
        )

    ran = correct = failed = skipped = 0
    started = time.time()

    total_cost = 0.0
    total_agent_stats = _empty_agent_stats()

    for i, case in enumerate(cases_to_run, start=1):
        case_id = case.get("id", str(i))
        label = f"[{i}/{len(cases_to_run)}] {case_id}"

        try:
            task_text = get_case_text(case)
            expected_answers = case.get("expected_answer") or []
            if not task_text or not expected_answers:
                case["evaluation"] = {"correct": None, "reason": "missing task text or expected_answer"}
                skipped += 1
                print(f"{label}: skipped")
                results.append(case)
                continue

            print(f"\n----- {label} ----- {task_text!r}")

            result = run_anytool(task_text, model=args.model)

            agent_stats = result.agent_stats
            agent_total = _sum_agent_stats(agent_stats)

            case_latency = agent_total["seconds"]
            case_prompt_tokens = agent_total["prompt_tokens"]
            case_completion_tokens = agent_total["completion_tokens"]
            case_cost = (
                case_prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
                + case_completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT_TOKENS
            )
            total_cost += case_cost
            for role in AGENT_ROLES:
                total_agent_stats[role]["seconds"] += agent_stats[role]["seconds"]
                total_agent_stats[role]["prompt_tokens"] += agent_stats[role]["prompt_tokens"]
                total_agent_stats[role]["completion_tokens"] += agent_stats[role]["completion_tokens"]

            evaluation = evaluate_case(expected_answers[0], result.answer if result.ok else None)
            if not result.ok:
                evaluation["reason"] = result.error
            evaluation["latency_seconds"] = round(case_latency, 3)
            evaluation["prompt_tokens"] = case_prompt_tokens
            evaluation["completion_tokens"] = case_completion_tokens
            evaluation["cost"] = case_cost
            evaluation["agent_stats"] = agent_stats
            case["evaluation"] = evaluation
            case["anytool_rounds_used"] = result.rounds_used

            ran += 1
            if evaluation["correct"]:
                correct += 1

            print(
                f"----- {label}: {'correct' if evaluation['correct'] else 'incorrect'}  "
                f"[{case_latency:.2f}s, ${case_cost:.6f}] -----"
            )
            print("      agent stats:")
            print(_format_agent_stats_block("        ", agent_stats, agent_total))

            results.append(case)

        except Exception as exc:  # noqa: BLE001 -- any failure skips this case only
            failed += 1
            case["evaluation"] = {
                "correct": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"{label}: failed ({type(exc).__name__}: {exc}) -> moving to next case")
            results.append(case)

        finally:
            # Save after every case (success, failure or skip) so progress is
            # never lost. `results` only ever grows: entries already saved
            # from earlier runs stay exactly as they were, and the new case
            # (whichever branch above ran) is appended, never overwriting
            # the file with just this run's cases.
            try:
                args.output.write_text(
                    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as write_exc:  # noqa: BLE001
                print(f"warning: could not save progress ({write_exc})")

    elapsed = time.time() - started
    total_agent_total = _sum_agent_stats(total_agent_stats)

    print("\n" + "=" * 60)
    print(f"Evaluated: {ran} / {len(cases_to_run)}  in {elapsed:.0f}s")
    if ran:
        print(f"Correct: {correct}/{ran}  ({100 * correct / ran:.1f}%)")
    if failed:
        print(f"Failed:  {failed}  (pipeline errors, skipped)")
    if skipped:
        print(f"Skipped: {skipped}  (missing task/expected_answer)")
    print(f"\nAgent stats (total across {ran} quer{'y' if ran == 1 else 'ies'}):")
    print(_format_agent_stats_block("  ", total_agent_stats, total_agent_total))
    print(f"\nTotal cost: ${total_cost:.6f}")
    print(f"Results written to: {args.output}  ({len(results)} total case(s) now saved there)")
    print("=" * 60)

    return ran, correct



if __name__ == "__main__":
    sys.exit(main())