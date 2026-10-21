#!/usr/bin/env python3
"""
Usage:
    export OPENAI_API_KEY=sk-...
    python run_query.py "Create a new circuit type named 'Wireless Backup' ..."
    python run_query.py --log-file results/my_run.log "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from anytool.pipeline import run_anytool  # noqa: E402

DEFAULT_LOG_FILE = Path(__file__).resolve().parent / "results" / "run_query_trace.log"


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
    parser.add_argument("query", help="The user request to resolve.")
    parser.add_argument("--model", default=None, help="OpenAI model (default: $ANYTOOL_MODEL or gpt-4o-mini).")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE,
                         help=f"Where to also save the full trace (default: {DEFAULT_LOG_FILE}).")
    args = parser.parse_args()

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    real_stdout = sys.stdout
    with args.log_file.open("w", encoding="utf-8") as log_file:
        sys.stdout = _Tee(real_stdout, log_file)
        try:
            print(f"(full trace is also being written to: {args.log_file})\n")

            result = run_anytool(args.query, model=args.model)

            print(f"\nrounds used: {result.rounds_used}")
            print(f"candidate pool ({len(result.candidate_pool)}): {result.candidate_pool}\n")

            if result.ok:
                print(json.dumps(result.answer, indent=2, ensure_ascii=False))
            else:
                print(f"FAILED: {result.error}")
        finally:
            sys.stdout = real_stdout

    if not result.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
