#!/usr/bin/env python3
"""Regenerate the 1-oracle/q cumulative-regret comparison plot + table.

Reads per-problem CSVs from four runs (all using Qwen2.5-Coder-7B-Instruct
on the full 1055 LCB problems, all using 1 oracle/q for evaluation):

  - Base 0-shot          (no training, no ICL)
  - ICL k=3 (stream)     (no training, kNN ICL anchors only)
  - RLOO + Nemotron-RM   (RM scores 200 rollouts/q for training, oracle only for eval)
  - ICL+SDFT (ours)      (1 oracle/q strict, online ICL+SDFT)

Each input CSV must have at least: step, problem_id, and one reward column
named one of {reward, pre_reward, greedy_reward}. We recompute cumulative
regret = Σ (1 − reward) so the curves are directly comparable.

Outputs:
  results/icl_sdft/cumreg_1oracle_methods.{png,pdf}
  results/icl_sdft/RESULTS_TABLE.md  (markdown summary)

Usage (from repo root):
  python icl_sdft/analyze_cumreg.py \\
      --base    results/path/to/base_results.csv \\
      --icl     results/path/to/icl_results.csv \\
      --rloo    results/path/to/rloo_results.csv \\
      --ours    results/icl_sdft/per_problem.csv \\
      --out_dir results/icl_sdft
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _reward_col(df):
    """Pick the first column that looks like a per-problem reward."""
    for c in ("greedy_reward", "pre_reward", "reward"):
        if c in df.columns:
            return c
    raise ValueError(f"no reward column in {df.columns.tolist()!r}")


def load_run(path):
    df = pd.read_csv(path).sort_values("step").reset_index(drop=True)
    df["R"] = df[_reward_col(df)]
    df["cumreg"] = (1.0 - df["R"]).cumsum()
    return df


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base 0-shot per-problem CSV")
    ap.add_argument("--icl",  required=True, help="ICL k=3 per-problem CSV")
    ap.add_argument("--rloo", required=True, help="RLOO+RM per-problem CSV")
    ap.add_argument("--ours", required=True, help="ICL+SDFT per-problem CSV")
    ap.add_argument("--out_dir", default="results/icl_sdft")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = load_run(args.base)
    icl  = load_run(args.icl)
    rloo = load_run(args.rloo)
    ours = load_run(args.ours)

    n = min(len(base), len(icl), len(rloo), len(ours))
    runs = [
        ("Base 0-shot",          base, "#888888", 0,   "0"),
        ("ICL k=3 (stream)",     icl,  "#ff7f0e", 0,   "0"),
        ("RLOO + Nemotron-RM",   rloo, "#2ca02c", 0,   "0 oracle / 200 RM"),
        ("ICL+SDFT (ours)",      ours, "#d62728", 1,   "0"),
    ]

    # Table
    table = ["| Method | avg R | cumreg | oracle/q (train) | RM calls/q (train) |",
             "|---|---:|---:|---:|---:|"]
    print(f"\n=== aligned on first {n} problems ===\n")
    print(f"{'method':<22}  {'avg_R':>7}  {'cumreg':>9}  {'tr_oracle':>10}  {'tr_RM':>20}")
    for label, df, color, tr_oracle, tr_rm in runs:
        avgR = df["R"].head(n).mean()
        cum  = df["cumreg"].head(n).iloc[-1]
        print(f"{label:<22}  {avgR:>7.4f}  {cum:>9.2f}  {tr_oracle:>10}  {tr_rm:>20}")
        table.append(f"| {label} | {avgR:.4f} | {cum:.2f} | {tr_oracle} | {tr_rm} |")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, df, color, tr_oracle, tr_rm in runs:
        lw = 2.6 if "ours" in label else 2.0
        ax.plot(df["step"].head(n), df["cumreg"].head(n),
                label=f"{label}  (avg R={df['R'].head(n).mean():.3f})",
                color=color, lw=lw)
    ax.set_xlabel("Problem index (step)")
    ax.set_ylabel(r"Cumulative regret  $\Sigma(1-R)$")
    ax.set_title(f"Online cumreg on LCB ({n} problems, Qwen2.5-Coder-7B, "
                 f"1 oracle/q for eval)")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    png = os.path.join(args.out_dir, "cumreg_1oracle_methods.png")
    pdf = os.path.join(args.out_dir, "cumreg_1oracle_methods.pdf")
    plt.savefig(png, dpi=150)
    plt.savefig(pdf)
    print(f"\nsaved: {png}")
    print(f"saved: {pdf}")

    table_md = os.path.join(args.out_dir, "RESULTS_TABLE.md")
    with open(table_md, "w") as f:
        f.write("# 1-oracle/q comparison on full LCB (n=" + str(n) + ")\n\n")
        f.write("\n".join(table) + "\n")
    print(f"saved: {table_md}")


if __name__ == "__main__":
    main()
