#!/usr/bin/env python3
"""
Sequential vs. Parallel execution time with confidence intervals.

Reads an .xlsx file with two sheets, "Sequential" and "Parallel", each with the
columns: Question, Command Match, Execution Status, Execution Time (s).
The number of devices is taken from the "Devices" column if it exists,
otherwise it is counted from the device names in the question ('sonic_4.4.0-N').

For every number of devices and every mode it uses only the questions that
executed successfully, and computes:
    mean, std dev, n, and the confidence interval  mean ± t(1-α/2, n-1) · s/√n

Then it draws a bar chart (mean) with error bars (CI) for 95% and 99%,
and tells for each group whether the two intervals overlap.

    pip install pandas openpyxl scipy matplotlib
    python3 ci_analysis.py Evaluation_CI_Analysis.xlsx
    python3 ci_analysis.py Evaluation_CI_Analysis.xlsx --devices 2 3 4 5 --levels 95 99
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MODES = ["Sequential", "Parallel"]
COLORS = {"Sequential": "#2A78D6", "Parallel": "#EB6834"}
DEVICE_NAME = re.compile(r"sonic_4\.4\.0-\d+")


def load_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    df = df[df["Question"].notna() & (df["Question"].astype(str).str.strip() != "")].copy()
    df["time"] = pd.to_numeric(df["Execution Time (s)"], errors="coerce")
    df["status"] = df["Execution Status"].astype(str).str.strip().str.lower()
    if "Devices" in df.columns and pd.to_numeric(df["Devices"], errors="coerce").notna().any():
        df["devices"] = pd.to_numeric(df["Devices"], errors="coerce")
    else:
        df["devices"] = df["Question"].astype(str).apply(lambda q: len(set(DEVICE_NAME.findall(q))))
    # keep only successful runs with a valid time
    df = df[(df["status"] == "success") & (df["time"] > 0)]
    df["mode"] = sheet
    return df[["mode", "devices", "time"]]


def confidence_interval(values, level):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return n, mean, float("nan"), float("nan")
    sd = values.std(ddof=1)
    margin = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / np.sqrt(n)
    return n, mean, sd, margin


def summarise(data, devices, levels):
    rows = []
    for level in levels:
        for d in devices:
            for mode in MODES:
                values = data[(data["mode"] == mode) & (data["devices"] == d)]["time"]
                n, mean, sd, margin = confidence_interval(values, level / 100)
                rows.append({
                    "Level (%)": level, "Devices": d, "Mode": mode, "n": n,
                    "Mean (s)": mean, "Std dev (s)": sd, "Margin (s)": margin,
                    "Lower (s)": mean - margin, "Upper (s)": mean + margin,
                })
    summary = pd.DataFrame(rows)

    # overlap check per level and device count
    verdicts = []
    for (level, d), g in summary.groupby(["Level (%)", "Devices"]):
        s = g[g["Mode"] == "Sequential"].iloc[0]
        p = g[g["Mode"] == "Parallel"].iloc[0]
        if s["Lower (s)"] > p["Upper (s)"]:
            verdict = "No overlap: Parallel is faster (significant)"
        elif p["Lower (s)"] > s["Upper (s)"]:
            verdict = "No overlap: Sequential is faster (significant)"
        else:
            verdict = "Overlap: cannot conclude"
        speedup = s["Mean (s)"] / p["Mean (s)"] if p["Mean (s)"] else float("nan")
        verdicts.append({"Level (%)": level, "Devices": d, "Speedup (x)": speedup, "Result": verdict})
    return summary, pd.DataFrame(verdicts)


def plot(summary, level, out_path):
    g = summary[summary["Level (%)"] == level]
    devices = sorted(g["Devices"].unique())
    x = np.arange(len(devices))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mode in enumerate(MODES):
        m = g[g["Mode"] == mode].set_index("Devices").loc[devices]
        bars = ax.bar(x + (i - 0.5) * width, m["Mean (s)"], width,
                      yerr=m["Margin (s)"], capsize=6, label=mode,
                      color=COLORS[mode], error_kw={"elinewidth": 1.2, "ecolor": "#333333"})
        for bar, mean, margin in zip(bars, m["Mean (s)"], m["Margin (s)"]):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + margin + 0.4, f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d} devices" for d in devices])
    ax.set_xlabel("Number of target devices")
    ax.set_ylabel("Mean execution time (s)")
    ax.set_title(f"Mean execution time with {level}% confidence intervals")
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color="#E6E6E6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Confidence intervals for Sequential vs. Parallel execution time.")
    parser.add_argument("file", type=Path, help="xlsx file with 'Sequential' and 'Parallel' sheets")
    parser.add_argument("--devices", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--levels", type=int, nargs="+", default=[95, 99])
    parser.add_argument("--outdir", type=Path, default=None, help="where to save charts and CSV (default: next to the file)")
    args = parser.parse_args()

    outdir = args.outdir or args.file.parent
    outdir.mkdir(parents=True, exist_ok=True)

    data = pd.concat([load_sheet(args.file, m) for m in MODES], ignore_index=True)
    summary, verdicts = summarise(data, args.devices, args.levels)

    pd.set_option("display.width", 160)
    for level in args.levels:
        print(f"\n===== {level}% confidence interval =====")
        print(summary[summary["Level (%)"] == level].drop(columns="Level (%)")
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print()
        print(verdicts[verdicts["Level (%)"] == level].drop(columns="Level (%)")
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        chart = outdir / f"ci_{level}.png"
        plot(summary, level, chart)
        print(f"\nChart saved: {chart}")

    csv_path = outdir / "ci_summary.csv"
    summary.merge(verdicts, on=["Level (%)", "Devices"]).to_csv(csv_path, index=False, float_format="%.3f")
    print(f"Table saved: {csv_path}")


if __name__ == "__main__":
    main()
