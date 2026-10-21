# SmartCLAI — Thesis Evaluation

Code, datasets and results used in the evaluation chapter of the master's thesis on **SmartCLAI**, an LLM-based assistant that turns natural-language requests into NetBox API calls and device configuration for SONiC switches.

## Repository layout

| Folder | What it contains |
|---|---|
| [`smartclai-evaluation/`](smartclai-evaluation/) | Evaluation of SmartCLAI's NetBox API selection (Agent 1: endpoint, method & parameters; Agent 2: request body) against the ground-truth dataset. |
| [`baselines/anytool/`](baselines/anytool/) | Re-implementation of the **AnyTool** baseline (Du, Wei & Zhang, 2024) on the same NetBox API pool, with its evaluation script and results. |
| [`baselines/reinvoke/`](baselines/reinvoke/) | Re-implementation of the **Re-Invoke** baseline (retrieval of tools via synthetic queries + embeddings), with its evaluation script. |
| [`parallel-vs-sequential/`](parallel-vs-sequential/) | Evaluation of parallel vs. sequential execution of device configuration: test questions, runner script, measured results (`.xlsx`) and the confidence-interval analysis. |

## Datasets

| File | Used by | Description |
|---|---|---|
| `smartclai-evaluation/ground_truth.json` | SmartCLAI | Natural-language requests with the expected NetBox endpoint, method, parameters and request body. |
| `baselines/anytool/data/ground_truth.json` | AnyTool | The same test cases in the format expected by the AnyTool evaluator. |
| `baselines/reinvoke/data/ground_truth.json` | Re-Invoke | The same test cases in the format expected by the Re-Invoke evaluator. |
| `baselines/anytool/endpoints.json` | AnyTool | The NetBox API pool (1,167 endpoints, 11 categories, 138 tools). |
| `baselines/reinvoke/data/tool_documents.json`, `synthetic_queries.json` | Re-Invoke | Tool documents and the synthetic queries used to build the retrieval index. |
| `parallel-vs-sequential/test_questions.csv` | Parallel vs. sequential | Configuration requests targeting 2–5 SONiC devices. |

## Requirements

- Python 3.10+
- An OpenAI API key for the baselines: copy `.env.example` to `.env` in the baseline folder and fill in your key. **Never commit `.env`.**
- The parallel/sequential runner needs a built SmartCLAI client binary and access to the lab devices (see that folder's README).

Each folder has its own README with the exact commands.
