"""AnyTool baseline (Du, Wei & Zhang, 2024 -- "AnyTool: Self-Reflective,
Hierarchical Agents for Large-Scale API Calls") applied to a NetBox API pool.

This package is a clean, standalone, Python-only re-implementation of the
baseline previously split across a Django service (prompt building) and a
C++ client (orchestration + OpenAI calls). Same prompts, same hierarchy,
same self-reflection mechanism -- just one language, in one place.
"""
