#!/usr/bin/env python3
"""
Usage:
    export OPENAI_API_KEY=sk-...
    python query.py
    # then type one request per line at the ">" prompt; blank line /
    # 'exit' / Ctrl-D quits. Edit k / model at the top of main() to
    # change those.

The index and the embedding model are both loaded ONCE, before the
first prompt -- loading the embedding model from disk is the
expensive part (it's a ~1.3GB local model), so running query.py as an
interactive loop instead of one process per request means only the
very first request in a session pays that cost; every request after
it reuses the same in-memory model and only pays for its own
embedding + comparison, which is fast.

Every request appends its full Step 1..6 breakdown (plus a
Timing/Tokens summary) to pipeline.log (in the project root), in
addition to printing it. sweep_k.py and evaluate.py append to this
SAME file too -- there's just the one log file for all three scripts.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from reinvoke import embedder, llm, retrieval  # noqa: E402
from reinvoke.prompts import INTENT_EXTRACTOR_PROMPT, REQUEST_BUILDER_PROMPT  # noqa: E402

INDEX_FILE = Path(__file__).resolve().parent / "data" / "index.pkl"
TOOL_DOCS_FILE = Path(__file__).resolve().parent / "data" / "tool_documents.json"

# One shared log file for query.py, sweep_k.py and evaluate.py -- all
# three are siblings in the project root, so this path is the same
# file no matter which of them opens it.
LOG_FILE = Path(__file__).resolve().parent / "pipeline.log"
_log_fh = open(LOG_FILE, "a", encoding="utf-8")


def log(message: str = "") -> None:
    """print(), plus append the same line to pipeline.log."""
    print(message)
    _log_fh.write(message + "\n")
    _log_fh.flush()


def run_header(title: str) -> None:
    """A timestamped separator so successive runs in the (appended)
    log file are easy to tell apart when scrolling back."""
    log("\n" + "=" * 90)
    log(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {title}")
    log("=" * 90)


# Loaded once per process and cached here -- the pickle on disk is ~10MB,
# so re-reading it from disk on every call would be wasteful. The first
# call actually opens data/index.pkl; every call after that (including
# from evaluate.py / sweep_k.py, which both import this instead of
# loading the pickle themselves) just returns the same in-memory object.
_tool_vectors_cache: dict | None = None


def get_tool_vectors() -> dict:
    """Returns the tool_vectors index, loading data/index.pkl from disk
    only the first time it's called."""
    global _tool_vectors_cache
    if _tool_vectors_cache is None:
        if not INDEX_FILE.exists():
            sys.exit(f"No index found at {INDEX_FILE}. Run build_index.py then convert_json_to_pkl.py first.")
        with open(INDEX_FILE, "rb") as f:
            _tool_vectors_cache = pickle.load(f)
    return _tool_vectors_cache


# gpt-4o-mini's own per-call token price (https://developers.openai.com/docs/models/gpt-4o-mini,
# used only to turn this run's token usage into a $ estimate.
# Embeddings are free: a local model, no API call, no tokens billed.

PRICE_PER_1M_INPUT_TOKENS = 0.15
PRICE_PER_1M_OUTPUT_TOKENS = 0.60


def extract_intents(user_query: str, model: str = "gpt-4o-mini") -> list[str]:

    prompt = INTENT_EXTRACTOR_PROMPT.format(user_query=user_query)
    # print(f"\nIntent extractor prompt (model={model}):\n{prompt}\n\n\n")
    reply = llm.complete(prompt, model=model, temperature=0.0)
    return [line.strip() for line in reply.splitlines() if line.strip()]


def _format_candidates(tool_documents: list[dict]) -> str:
    """Renders the retriever's top-k tool_documents as a numbered list for
    REQUEST_BUILDER_PROMPT, each with its own schema, best match (rank 1)
    first."""
    blocks = []
    for i, doc in enumerate(tool_documents, start=1):
        blocks.append(
            f"Candidate {i} ({doc['endpoint_id']}):\n"
            f"  method: {doc['method']}\n"
            f"  path: {doc['path']}\n"
            f"  parameters: {json.dumps(doc.get('parameters', {}))}\n"
            f"  request_body: {json.dumps(doc.get('request_body', {}))}"
        )
    return "\n\n".join(blocks)


def build_request(user_query: str, tool_documents: list[dict], model: str = "gpt-4o-mini") -> dict:

    prompt = REQUEST_BUILDER_PROMPT.format(
        candidates=_format_candidates(tool_documents),
        user_query=user_query,
    )
    print(f"\nRequest builder prompt (model={model}):\n{prompt}\n\n\n")

    reply = llm.complete(prompt, model=model, temperature=0.0)
    reply = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(reply)


def _usage_delta(before: dict, after: dict) -> dict:
    return {
        "calls": after["calls"] - before["calls"],
        "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        "total_tokens": after["total_tokens"] - before["total_tokens"],
    }


def run_query(
    request: str,
    tool_vectors: dict,
    k: int = 4,
    model: str = "gpt-4o-mini",
    embedding_model: str = embedder.DEFAULT_EMBEDDING_MODEL,
) -> None:

    log(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    log(f"Query: {request}\n")

    # ---- step 2: intent extraction ------------------------------------
    usage_before_intent = llm.token_usage_totals()
    started_time_intent = time.time()
    intents = extract_intents(request, model=model)
    time_intent = time.time() - started_time_intent
    usage_after_intent = llm.token_usage_totals()

    # ---- step 3: embed each intent -------------------------------------
    # (the embedding model itself is already loaded -- see main() -- so
    # this only pays for encoding these few short strings)
    started_time_retriever = time.time()
    intent_vectors = embedder.embed_texts(intents, model=embedding_model, is_query=True)

    # ---- step 4 & 5: compare each intent against every tool, merge top k ----
    ranked = retrieval.rank_tools(intent_vectors, tool_vectors)
    top_results = retrieval.top_k(ranked, k)
    time_retriever = time.time() - started_time_retriever

    # ---- step 6: build the actual call, letting the agent pick from the top k ----
    tool_documents = {doc["endpoint_id"]: doc for doc in json.loads(TOOL_DOCS_FILE.read_text())}
    candidates = [tool_documents[eid] for eid, _, _ in top_results if eid in tool_documents]

    built = None
    build_error = None
    started_time_build_req = time.time()
    usage_before_build = llm.token_usage_totals()
    if candidates:
        try:
            built = build_request(request, candidates, model=model)
        except Exception as exc:  # noqa: BLE001 -- any pipeline failure is logged, not a crash
            build_error = exc
    time_build = time.time() - started_time_build_req
    usage_after_build = llm.token_usage_totals()

    # ---- print/log Steps 2-6, then Timing + Tokens ----------------------
    log(f"\nStep 2 - Intent extraction (took {time_intent:.2f}s):")
    for intent in intents:
        log(f"  - {intent}")

    log(f"\nStep 3 - Embedding each intent (model={embedding_model}, took {time_retriever:.2f}s):")
    for intent, vector in zip(intents, intent_vectors):
        preview = [round(v, 4) for v in vector[:5]]
        log(f"  '{intent}' -> {len(vector)}-dim vector, first 5 values: {preview}")

    log(f"\nStep 4 - Comparing each intent against all {len(tool_vectors)} tool(s) (top 10 shown per intent):")
    for i, (intent, vector) in enumerate(zip(intents, intent_vectors), start=1):
        scores = retrieval.similarity_scores(vector, tool_vectors)
        ranks = retrieval.rank_within_intent(scores)
        log(f"  Intent {i}: '{intent}'")
        top10 = sorted(scores, key=lambda eid: scores[eid], reverse=True)[:10]
        for eid in top10:
            log(f"      {eid:<45} score={scores[eid]:.4f}  rank={ranks[eid]}/{len(tool_vectors)}")

    log(f"\nStep 5 - Final merged top {k}:")
    for endpoint_id, rank, score in top_results:
        log(f"  {endpoint_id:<45} best_rank={rank}  best_score={score:.4f}")

    log(f"\nStep 6 - Building the actual call (chosen from top {len(candidates)} candidate(s), took {time_build:.2f}s):")
    if build_error is not None:
        log(f"  failed: {build_error}")
    elif built is not None:
        log("  " + json.dumps(built, indent=2).replace("\n", "\n  "))
    else:
        log(f"  none of the top {k} endpoint(s) were found in tool_documents.json")

    intent_usage = _usage_delta(usage_before_intent, usage_after_intent)
    build_usage = _usage_delta(usage_before_build, usage_after_build)
    total_prompt = intent_usage["prompt_tokens"] + build_usage["prompt_tokens"]
    total_completion = intent_usage["completion_tokens"] + build_usage["completion_tokens"]
    total_tokens = intent_usage["total_tokens"] + build_usage["total_tokens"]
    cost = (
        total_prompt / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
        + total_completion / 1_000_000 * PRICE_PER_1M_OUTPUT_TOKENS
    )

    log("\nTiming:")
    log(f"  Intent extraction: {time_intent:.2f}s")
    log(f"  Retriever:         {time_retriever:.2f}s")
    log(f"  Build request:     {time_build:.2f}s")
    log(f"  Total:             {time_intent + time_retriever + time_build:.2f}s")
    log("\nTokens (detailed):")
    log(
        f"  Intent extraction: {intent_usage['prompt_tokens']} prompt + "
        f"{intent_usage['completion_tokens']} completion = {intent_usage['total_tokens']} total  "
        f"({intent_usage['calls']} call(s))"
    )
    log(
        f"  Build request:     {build_usage['prompt_tokens']} prompt + "
        f"{build_usage['completion_tokens']} completion = {build_usage['total_tokens']} total  "
        f"({build_usage['calls']} call(s))"
    )
    log(f"  Total Tokens:          = {total_tokens} total")
    log(f"  Cost:              ${cost:.6f}")


def main() -> None:
    k = 4
    model = "gpt-4o-mini"
    embedding_model = embedder.DEFAULT_EMBEDDING_MODEL

    run_header(f"query.py (interactive)  k={k}  model={model}")

    # ---- load the index once -- cached in memory for the whole session ----
    tool_vectors = get_tool_vectors()
    log(f"Step 1 - Index loaded: {len(tool_vectors)} tool(s)")

    # ---- load the embedding model once too, right here, instead of
    # letting the first request pay for it inside the "Retriever" timing.
    # After this, every request below -- no matter how many you run in
    # this session -- reuses the same in-memory model and only pays for
    # its own embedding + comparison. ----
    log(f"\nLoading embedding model ({embedding_model})...")
    started_time_model = time.time()
    embedder.get_model(embedding_model)
    log(f"Model ready in {time.time() - started_time_model:.2f}s -- stays loaded for the rest of this session.\n")

    print("Enter a request and press Enter to run it through the pipeline.")
    print("Blank line, 'exit', or Ctrl-D to quit.\n")

    while True:
        try:
            request = input("> ").strip()
        except EOFError:
            print()
            break
        if not request or request.lower() in {"exit", "quit"}:
            break
        run_query(request, tool_vectors, k=k, model=model, embedding_model=embedding_model)


if __name__ == "__main__":
    sys.exit(main())
