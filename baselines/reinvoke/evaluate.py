#!/usr/bin/env python3
"""
Usage:
    export OPENAI_API_KEY=sk-...
    python evaluate.py

Appends to the SAME pipeline.log (in the project root) that query.py
and sweep_k.py use -- one shared log file for all three scripts. Runs
many ground-truth cases in one go, so instead of a per-case breakdown
it logs one concise average at the end: Intent extraction / Retriever
/ Build request / Total (each averaged across every case that ran),
plus averaged tokens and cost.
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
OUTPUT_FILE = Path(__file__).resolve().parent / "results" / "evaluation.json"
SUMMARY_OUTPUT_FILE = Path(__file__).resolve().parent / "results" / "evaluation_summary.json"


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
    cost -- called once at the end, never per case."""
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
    limit = None  # set to an int to only run the first N cases, e.g. 20 for a quick dry run
    k = 4  # how many top-ranked candidates retrieval hands the agent
    model = "gpt-4o-mini"
    embedding_model = embedder.DEFAULT_EMBEDDING_MODEL

    run_header(f"evaluate.py  limit={limit}  k={k}  model={model}")

    tool_vectors = get_tool_vectors()  # loaded once, cached in memory (see query.py)

    tool_documents = {doc["endpoint_id"]: doc for doc in json.loads(TOOL_DOCS_FILE.read_text())}

    cases = json.loads(GROUND_TRUTH_FILE.read_text(encoding="utf-8"))
    if limit:
        cases = cases[:limit]

    log(f"Loaded {len(cases)} case(s) from {GROUND_TRUTH_FILE}\n")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    ran = correct = failed = 0
    started = time.time()

    # Summed across every case that actually ran an LLM call, so the end
    # summary can report per-query averages the same way the paper's
    # Table II/VI do (Average Token Usage/1Q, Cost ($/100Q), Latency (s)).
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    intent_times: list[float] = []
    retriever_times: list[float] = []
    build_times: list[float] = []
    total_times: list[float] = []

    for i, case in enumerate(cases, start=1):
        case_id = case.get("id", str(i))
        task_text = get_case_text(case)
        label = f"[{i}/{len(cases)}] {case_id}"

        expected_answers = case.get("expected_answer") or []
        if not task_text or not expected_answers:
            case["evaluation"] = {"correct": None, "reason": "missing task text or expected_answer"}
            log(f"{label}: skipped")
            continue

        usage_before = llm.token_usage_totals()
        time_intent = time_retriever = time_build = 0.0
        error = None
        actual = None
        top_results: list = []
        try:
            time_intent_start = time.time()
            try:
                intents = extract_intents(task_text, model=model)
            finally:
                time_intent = time.time() - time_intent_start

            time_retriever_start = time.time()
            try:
                intent_vectors = embedder.embed_texts(intents, model=embedding_model, is_query=True)
                ranked = retrieval.rank_tools(intent_vectors, tool_vectors)
                top_results = retrieval.top_k(ranked, k)
            finally:
                time_retriever = time.time() - time_retriever_start
            candidates = [tool_documents[eid] for eid, _, _ in top_results if eid in tool_documents]
            if not candidates:
                raise RuntimeError(f"none of the top {k} endpoint(s) were found in tool_documents.json")

            time_build_start = time.time()
            try:
                actual = build_request(task_text, candidates, model=model)
            finally:
                time_build = time.time() - time_build_start
        except Exception as exc:  # noqa: BLE001 -- any pipeline failure counts as "failed", not a crash
            error = exc

        case_latency = time_intent + time_build

        usage_after = llm.token_usage_totals()
        case_prompt_tokens = usage_after["prompt_tokens"] - usage_before["prompt_tokens"]
        case_completion_tokens = usage_after["completion_tokens"] - usage_before["completion_tokens"]
        case_cost = (
            case_prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
            + case_completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT_TOKENS
        )
        total_prompt_tokens += case_prompt_tokens
        total_completion_tokens += case_completion_tokens
        total_cost += case_cost
        intent_times.append(time_intent)
        retriever_times.append(time_retriever)
        build_times.append(time_build)
        total_times.append(case_latency)

        if error is not None:
            failed += 1
            case["evaluation"] = {
                "correct": False, "reason": str(error),
                "latency_seconds": round(case_latency, 3),
                "prompt_tokens": case_prompt_tokens, "completion_tokens": case_completion_tokens,
                "cost": case_cost,
            }
            log(f"{label}: failed ({error})  [{case_latency:.2f}s, ${case_cost:.6f}]")
            OUTPUT_FILE.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
            continue

        # The agent may have picked any of the k candidates, not
        # necessarily retrieval's #1
        chosen_endpoint_id = actual.get("endpoint_id") or top_results[0][0]

        evaluation = evaluate_case(expected_answers[0], actual)
        evaluation["candidate_endpoint_ids"] = [eid for eid, _, _ in top_results]
        evaluation["retrieved_endpoint_id"] = chosen_endpoint_id
        evaluation["retrieved_method"] = actual["method"]
        evaluation["retrieved_path"] = actual["endpoint"]
        evaluation["retrieved_body"] = actual["body"]
        evaluation["latency_seconds"] = round(case_latency, 3)
        evaluation["prompt_tokens"] = case_prompt_tokens
        evaluation["completion_tokens"] = case_completion_tokens
        evaluation["cost"] = case_cost
        case["evaluation"] = evaluation

        ran += 1
        if evaluation["correct"]:
            correct += 1

        status = "correct" if evaluation["correct"] else "incorrect"
        log(f"{label}: {status}  (chosen: {chosen_endpoint_id})  [{case_latency:.2f}s, ${case_cost:.6f}]")

        # Save after every case so an interruption never loses progress.
        OUTPUT_FILE.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - started
    denom = ran + failed  # every case that actually made an LLM call, correct or not
    avg_intent_time = sum(intent_times) / denom
    avg_retriever_time = sum(retriever_times) / denom
    avg_build_time = sum(build_times) / denom
    avg_latency = sum(total_times) / denom
    avg_prompt_tokens = total_prompt_tokens / denom
    avg_completion_tokens = total_completion_tokens / denom
    avg_cost = total_cost / denom

    log("\n" + "=" * 60)
    log(f"Evaluated: {ran} / {len(cases)}  in {elapsed:.0f}s")
    if ran:
        log(f"Correct: {correct}/{ran}  ({100 * correct / ran:.1f}%)")
    if failed:
        log(f"Failed:  {failed}  (pipeline error, e.g. a dropped OpenAI call)")
    log(f"\nResults written to: {OUTPUT_FILE}")
    log("=" * 60)

    log_average_summary(
        label="evaluate.py",
        n=denom,
        avg_intent_time=avg_intent_time,
        avg_retriever_time=avg_retriever_time,
        avg_build_time=avg_build_time,
        avg_total_time=avg_latency,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_completion_tokens=avg_completion_tokens,
        avg_cost=avg_cost,
        total_cost=total_cost,
    )

    summary = {
        "k": k,
        "ran": ran,
        "failed": failed,
        "total_cases": len(cases),
        "accuracy": (correct / ran),
        "avg_intent_latency_seconds": avg_intent_time,
        "avg_retriever_latency_seconds": avg_retriever_time,
        "avg_build_latency_seconds": avg_build_time,
        "avg_input_tokens_per_query": avg_prompt_tokens,
        "avg_output_tokens_per_query": avg_completion_tokens,
        "avg_latency_seconds_per_query": avg_latency,
        "avg_cost_per_query": avg_cost,
        "total_cost": total_cost,
        "elapsed_seconds": round(elapsed, 1),
    }
    SUMMARY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Summary written to: {SUMMARY_OUTPUT_FILE}")


if __name__ == "__main__":
    sys.exit(main())
