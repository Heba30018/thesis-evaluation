#!/usr/bin/env python3
"""
Usage:
    export OPENAI_API_KEY=sk-...
    python sweep_k.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from reinvoke import embedder, llm, retrieval  # noqa: E402
from reinvoke.evaluation import evaluate_case, get_case_text  # noqa: E402
from query import (  # noqa: E402
    PRICE_PER_1M_INPUT_TOKENS,
    PRICE_PER_1M_OUTPUT_TOKENS,
    build_request,
    extract_intents,
    get_tool_vectors,
    log,
    run_header,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.json"
INDEX_FILE = DATA_DIR / "index.pkl"
TOOL_DOCS_FILE = DATA_DIR / "tool_documents.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "results" / "k_sweep.json"


def log_average_summary(
    label: str,
    n: int,
    avg_intent_time: float,
    avg_retriever_time: float,
    avg_build_time: float,
    avg_total_time: float,
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    avg_cost: float,
    total_cost: float,
) -> None:
    """Concise block: Intent extraction / Retriever / Build request /
    Total (each averaged over n cases), then averaged tokens and
    cost -- called once per K, never per case."""
    log(f"\n{label} -- averages over {n} case(s):")
    log(f"  Intent extraction: {avg_intent_time:.2f}s")
    log(f"  Retriever:         {avg_retriever_time:.2f}s")
    log(f"  Build request:     {avg_build_time:.2f}s")
    log(f"  Total:             {avg_total_time:.2f}s")
    log(
        f"  Tokens (avg/query): {avg_prompt_tokens:.1f} prompt + "
        f"{avg_completion_tokens:.1f} completion = {avg_prompt_tokens + avg_completion_tokens:.1f} total"
    )
    log(f"  Cost: ${avg_cost:.6f}/query   (total: ${total_cost:.6f} for {n} query(ies))")


def main() -> None:
    k_values = [1, 3, 4, 5, 6, 8, 10]
    sample_size = 30  # use the first 30 questions in GROUND_TRUTH_FILE
    model = "gpt-4o-mini"
    embedding_model = embedder.DEFAULT_EMBEDDING_MODEL

    run_header(f"sweep_k.py  sample_size={sample_size}  K values={k_values}  model={model}")

    tool_vectors = get_tool_vectors()  # loaded once, cached in memory (see query.py)

    tool_documents = {doc["endpoint_id"]: doc for doc in json.loads(TOOL_DOCS_FILE.read_text())}

    all_cases = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    cases = all_cases[:sample_size]
    log(f"Sample: first {len(cases)} question(s) from {GROUND_TRUTH_FILE}")
    log(f"K values: {k_values}\n")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = {"sample_size": 0, "by_k": {}}

    for k in k_values:
        log(f"\n--- K = {k} ---")
        correct = failed = 0
        intent_times: list[float] = []
        retriever_times: list[float] = []
        build_times: list[float] = []
        total_times: list[float] = []
        prompt_tokens_list: list[int] = []
        completion_tokens_list: list[int] = []
        costs: list[float] = []
        per_case = []

        # ---- one single loop: every case goes end to end through intent
        # extraction -> retriever -> request building, right here. ----
        for i, case in enumerate(cases, start=1):
            case_id = case.get("id", str(i))
            task_text = get_case_text(case)
            expected_answers = case.get("expected_answer") or []
            if not task_text or not expected_answers:
                log(f"[{i}/{len(cases)}] {case_id}: skipped (missing task text or expected_answer)")
                continue

            usage_before_intent = llm.token_usage_totals()
            started_time_intent = time.time()
            intents = extract_intents(task_text, model=model)
            time_intent = time.time() - started_time_intent
            usage_after_intent = llm.token_usage_totals()

            started_time_retriever = time.time()
            intent_vectors = embedder.embed_texts(intents, model=embedding_model, is_query=True)
            ranked = retrieval.rank_tools(intent_vectors, tool_vectors)
            top_results = retrieval.top_k(ranked, k)
            time_retriever = time.time() - started_time_retriever
            candidates = [tool_documents[eid] for eid, _, _ in top_results if eid in tool_documents]

            error = None
            actual = None
            usage_before_build = llm.token_usage_totals()
            started_time_build_req = time.time()
            try:
                if not candidates:
                    raise RuntimeError(f"none of the top {k} endpoint(s) were found in tool_documents.json")
                actual = build_request(task_text, candidates, model=model)
            except Exception as exc:  # noqa: BLE001 -- any pipeline failure counts as "failed", not a crash
                error = exc
            time_build = time.time() - started_time_build_req
            usage_after_build = llm.token_usage_totals()

            case_prompt_tokens = (
                (usage_after_intent["prompt_tokens"] - usage_before_intent["prompt_tokens"])
                + (usage_after_build["prompt_tokens"] - usage_before_build["prompt_tokens"])
            )
            case_completion_tokens = (
                (usage_after_intent["completion_tokens"] - usage_before_intent["completion_tokens"])
                + (usage_after_build["completion_tokens"] - usage_before_build["completion_tokens"])
            )
            case_cost = (
                case_prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
                + case_completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT_TOKENS
            )
            case_total_time = time_intent  + time_build

            intent_times.append(time_intent)
            retriever_times.append(time_retriever)
            build_times.append(time_build)
            total_times.append(case_total_time)
            prompt_tokens_list.append(case_prompt_tokens)
            completion_tokens_list.append(case_completion_tokens)
            costs.append(case_cost)

            if error is not None:
                failed += 1
                per_case.append({
                    "case_id": case_id, "correct": False, "reason": str(error),
                    "latency_seconds": round(case_total_time, 3), "cost": case_cost,
                })
                log(
                    f"  [{i}/{len(cases)}] {case_id}: failed ({error})  "
                    f"[{case_total_time:.2f}s, {case_prompt_tokens + case_completion_tokens} tokens, ${case_cost:.6f}]"
                )
                continue

            evaluation = evaluate_case(expected_answers[0], actual)
            evaluation["latency_seconds"] = round(case_total_time, 3)
            evaluation["cost"] = case_cost
            per_case.append({"case_id": case_id, **evaluation})
            if evaluation["correct"]:
                correct += 1
            status = "correct" if evaluation["correct"] else "incorrect"
            log(
                f"  [{i}/{len(cases)}] {case_id}: {status}  "
                f"[{case_total_time:.2f}s, {case_prompt_tokens + case_completion_tokens} tokens, ${case_cost:.6f}]"
            )

        # ---- loop over every case for this K is done -- now average ----
        ran = len(costs)  # every case that actually ran (correct + incorrect + failed)
        false_count = ran - correct  # "false" = incorrect OR failed, same convention as before
        accuracy = correct / ran if ran else 0.0
        avg_intent_time = sum(intent_times) / ran
        avg_retriever_time = sum(retriever_times) / ran
        avg_build_time = sum(build_times) / ran
        avg_latency = sum(total_times) / ran
        avg_prompt_tokens = sum(prompt_tokens_list) / ran
        avg_completion_tokens = sum(completion_tokens_list) / ran
        avg_cost = sum(costs) / ran
        total_cost = sum(costs)
        results["sample_size"] = ran
        results["by_k"][str(k)] = {
            "correct": correct,
            "false": false_count,
            "ran": ran,
            "failed": failed,
            "accuracy": accuracy,
            "avg_latency_seconds": avg_latency,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "cases": per_case,
        }
        log(
            f"K={k}: {correct} true / {false_count} false  ({100 * accuracy:.1f}% accuracy)   "
            f"avg latency={avg_latency:.2f}s/query   avg cost=${avg_cost:.6f}/query"
        )
        log_average_summary(
            label=f"K={k}",
            n=ran,
            avg_intent_time=avg_intent_time,
            avg_retriever_time=avg_retriever_time,
            avg_build_time=avg_build_time,
            avg_total_time=avg_latency,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_completion_tokens=avg_completion_tokens,
            avg_cost=avg_cost,
            total_cost=total_cost,
        )

        # Save after every K so an interruption never loses progress.
        OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        # ---- then (and only then) move on to the next K -----------------

    # ---- summary ----------------------------------------------------------
    log("\n" + "=" * 78)
    log(f"{'K':>4}  {'true':>5}  {'false':>6}  {'accuracy':>9}  {'avg latency':>12}  {'avg cost':>11}")
    best_k, best_accuracy = None, -1.0
    for k in k_values:
        row = results["by_k"][str(k)]
        log(
            f"{k:>4}  {row['correct']:>5}  {row['false']:>6}  {100 * row['accuracy']:>8.1f}%  "
            f"{row['avg_latency_seconds']:>10.2f}s  ${row['avg_cost']:>9.6f}"
        )
        if row["accuracy"] > best_accuracy:
            best_k, best_accuracy = k, row["accuracy"]
    log("=" * 78)
    log(f"Best K: {best_k}  ({100 * best_accuracy:.1f}% accuracy)")
    log(f"Full results written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    sys.exit(main())
