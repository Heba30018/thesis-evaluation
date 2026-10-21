# Re-Invoke Baseline

Re-implementation of the **Re-Invoke** tool-retrieval baseline on the NetBox API pool. Every tool (endpoint) is represented by the average embedding of its document combined with synthetic queries; a user request is embedded and the closest tools are retrieved and passed to the LLM.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
```

## 1. Build the retrieval index (one time)
The index files are not stored in the repository because they are large. Rebuild them with:
```bash
python build_index.py          # embeds all 1,167 endpoints -> data/index.json
python convert_json_to_pkl.py  # data/index.json -> data/index.pkl (used by the scripts)
```
Embeddings use a local sentence-transformers model (`BAAI/bge-large-en-v1.5`), downloaded on the first run; no API key is needed for this step.

## 2. Run
```bash
python query.py "Create a new device named ..."   # single request
python evaluate.py                                 # full evaluation against data/ground_truth.json
python sweep_k.py                                  # evaluation for different top-k values
```

## Files
| File | Purpose |
|---|---|
| `reinvoke/embedder.py` | Embedding model wrapper |
| `reinvoke/retrieval.py` | Top-k tool retrieval |
| `reinvoke/prompts.py`, `reinvoke/llm.py` | Prompt and OpenAI call |
| `reinvoke/evaluation.py` | Comparison with the ground truth |
| `data/` | Ground truth, tool documents and synthetic queries |
