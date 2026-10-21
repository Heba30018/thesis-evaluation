# Parallel vs. Sequential Execution

Measures how much faster SmartCLAI configures several SONiC devices when the SSH sessions run **in parallel** instead of **one device after another**.

## Files
| File | Purpose |
|---|---|
| `test_questions.csv` | The configuration requests (2–5 target devices each) |
| `evaluation_parallelization.py` | Runs every request through the SmartCLAI client and records Command Match, Execution Status and Execution Time |
| `Evaluation_Parallel_vs_Sequential.xlsx` | Measured results — tab **Sequential** and tab **Parallel** (one row per request, with the number of devices) |
| `ci_analysis.py` | Mean execution time and 95% / 99% confidence intervals per number of devices, with bar charts (error bars = CI) |
| `ci_analysis_colab.ipynb` | Same analysis as a Google Colab notebook |

## 1. Run the measurements
Run from the SmartCLAI client repository after building it (needs the lab devices). SSH pre-warming is disabled in both modes so connection setup is included in the measured time.
```bash
python3 evaluation_parallelization.py --mode sequential --input test_questions.csv   # 1 SSH worker
python3 evaluation_parallelization.py --mode parallel   --input test_questions.csv   # parallel workers
```
Results are written to `Evaluation_sequential_results.csv` / `Evaluation_parallel_results.csv`.

## 2. Statistical analysis
```bash
pip install pandas openpyxl scipy matplotlib
python3 ci_analysis.py Evaluation_Parallel_vs_Sequential.xlsx --levels 95 99
```
Only successfully executed requests are used. For each number of devices the script reports n, mean, standard deviation and the confidence interval `mean ± t(1-α/2, n-1)·s/√n`, and whether the sequential and parallel intervals overlap.

Or open `ci_analysis_colab.ipynb` in Google Colab, run all cells and upload the `.xlsx` when asked.
