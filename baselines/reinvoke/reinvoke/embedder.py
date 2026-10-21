"""Turns text into vectors (via a local sentence-transformers model, no
API call needed), and turns several vectors for the same tool into the
one averaged vector that represents it in the index.

Default model is BAAI/bge-large-en-v1.5, downloaded once from Hugging
Face and cached on disk (~/.cache/huggingface) -- after that first
download, embedding runs entirely offline, with no API key and no
network call, unlike the OpenAI embeddings this replaced.

BGE's own model card recommends prepending a fixed instruction to a
QUERY (never to a passage/document) for best retrieval quality:
"no instruction only has a slight degradation in retrieval performance",
so it's optional but worth doing. See
https://huggingface.co/BAAI/bge-large-en-v1.5
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_models: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Loads (and caches in-process) the sentence-transformers model, so
    a single run only pays the load cost once no matter how many texts
    get embedded, or how many times this is called. Public (no leading
    underscore) because query.py calls this directly to force the model
    to load before the first request, instead of waiting for it to load
    lazily inside the first embed_texts() call."""
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def embed_texts(
    texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL, is_query: bool = False
) -> list[list[float]]:
    """Returns one L2-normalized embedding vector per input text, in the
    same order. Pass is_query=True when embedding a user query/intent
    (adds BGE's recommended query instruction); leave it False (the
    default) for tool/passage text, which gets no instruction."""
    st_model = get_model(model)
    if is_query:
        texts = [QUERY_INSTRUCTION + t for t in texts]
    vectors = st_model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def average_vectors(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean of several equal-length vectors."""
    return np.mean(np.array(vectors), axis=0).tolist()
