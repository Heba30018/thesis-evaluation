#!/usr/bin/env python3
"""Builds the retrieval index: for every tool document, pairs its already-
generated synthetic queries with the document text, embeds the resulting
augmented copies, and averages them into one vector per tool. Saves the
vectors so the embedding calls never need to be repeated for a tool
that's already indexed.

data/tool_documents.json and data/synthetic_queries.json are both
ready-made input data for this script -- it only reads them, it never
generates or rewrites either one.

Embeddings come from a local sentence-transformers model
(BAAI/bge-large-en-v1.5 by default -- see reinvoke/embedder.py), not an
API call, so no OPENAI_API_KEY is needed to run this script. The model
weights (~1.3GB) are downloaded from Hugging Face the first time it
runs and cached on disk after that.

Usage:
    python build_index.py --limit 5          # a small pilot run
    python build_index.py                     # all 1,167 endpoints
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from reinvoke import embedder  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
TOOL_DOCS_FILE = DATA_DIR / "tool_documents.json"
QUERIES_FILE = DATA_DIR / "synthetic_queries.json"
INDEX_FILE = DATA_DIR / "index.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N tools.")
    parser.add_argument("--embedding-model", default=embedder.DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    if not TOOL_DOCS_FILE.exists():
        sys.exit(f"No tool documents found at {TOOL_DOCS_FILE}.")
    tool_documents = json.loads(TOOL_DOCS_FILE.read_text())
    if args.limit:
        tool_documents = tool_documents[: args.limit]

    queries_by_tool = load_json(QUERIES_FILE)
    index = load_json(INDEX_FILE)

    print(f"Indexing {len(tool_documents)} tool(s)\n")

    for i, tool_document in enumerate(tool_documents, start=1):
        endpoint_id = tool_document["endpoint_id"]
        if endpoint_id in index:
            print(f"[{i}/{len(tool_documents)}] {endpoint_id}: already indexed")
            continue

        queries = queries_by_tool.get(endpoint_id)

        tool_text = json.dumps(tool_document, indent=2)
        augmented_documents = [f"Documentation: {tool_text}\nQuery: {query}" for query in queries]
        vectors = embedder.embed_texts(augmented_documents, model=args.embedding_model)
        index[endpoint_id] = embedder.average_vectors(vectors)
        INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False))

        print(f"[{i}/{len(tool_documents)}] {endpoint_id}: indexed")



if __name__ == "__main__":
    sys.exit(main())
