# SmartCLAI — NetBox API Selection Evaluation

Evaluates SmartCLAI's two agents against the ground-truth dataset:
- **Agent 1** – NetBox endpoint, HTTP method and parameters
- **Agent 2** – request body

| File | Purpose |
|---|---|
| `ground_truth.json` | Test requests with the expected endpoint, method, parameters and body |
| `netbox_data_100.json` | NetBox data used by the test requests |
| `run_batch_eval.py` | Sends every request through the SmartCLAI client (`--netbox-ui-bridge`) and records the agents' output, timing and token usage |
| `evaluate_agents.py` | Compares the recorded output with the ground truth |

## Run
These scripts drive the SmartCLAI client binary, so run them from the SmartCLAI client repository after building it:
```bash
python3 run_batch_eval.py --binary ../build/smartclai --ground-truth ground_truth.json
python3 evaluate_agents.py --ground-truth ground_truth.json --actual <output-of-run_batch_eval>.json
```
