#!/usr/bin/env python3
"""Converts data/index.json (built by build_index.py) into data/index.pkl.

Usage:
    python convert_json_to_pkl.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_JSON = DATA_DIR / "index.json"
INDEX_PKL = DATA_DIR / "index.pkl"


def main() -> None:
    if not INDEX_JSON.exists():
        sys.exit(f"No index found at {INDEX_JSON}. Run build_index.py first.")

    index = json.loads(INDEX_JSON.read_text())
    index = {endpoint_id: np.array(vector) for endpoint_id, vector in index.items()}

    with open(INDEX_PKL, "wb") as f:
        pickle.dump(index, f)

    print(f"Converted {len(index)} vector(s): {INDEX_JSON} -> {INDEX_PKL}")


if __name__ == "__main__":
    sys.exit(main())
