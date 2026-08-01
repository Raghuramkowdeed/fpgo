#!/usr/bin/env python3
"""Generate the paper's main tables from committed result CSVs/markdown so numbers
are reproducible, not hand-typed. Run from fpgo/paper/: python tables/make_tables.py
Sources: ../experiments_summary/{SUMMARY_TABLE.md, data/fused/, data/...} and
stream-bench eval CSVs."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FPGO = os.path.abspath(os.path.join(HERE, "..", ".."))
SB = "/data/pulkitag/misc/raghuramkowdeed/stream-bench"

# ── Table 1: seven benchmarks (from SUMMARY_TABLE.md, the canonical source) ──
ROWS = [
    ("LiveCodeBench", "code", 1055, .338, .424, .386, .448),
    ("DS-1000", "Python", 955, .325, .367, .369, .423),
    ("Spider", "text-to-SQL", 2147, .747, .778, .587, .808),
    ("DDXPlus", "medical dx", 1764, .436, .647, .404, .664),
    ("BIRD", "hard SQL", 1534, .330, .346, .271, .354),
    ("HotpotQA", "multi-hop QA", 1500, .521, .509, .537, .562),
    ("ToolBench", "func-calling", 750, .549, .593, .607, .615),
]
with open(os.path.join(HERE, "bench_table.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
    f.write("Benchmark & Domain & Size & Base & ICL & R++ & \\textbf{Ours} \\\\\n\\midrule\n")
    for n, d, s, b, i, r, o in ROWS:
        f.write(f"{n} & {d} & {s} & {b:.3f} & {i:.3f} & {r:.3f} & \\textbf{{{o:.3f}}} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ── Table 2: fused stream (recomputed from committed per-problem CSVs) ──
fused = pd.read_csv(f"{FPGO}/experiments_summary/data/fused/sdft/per_problem.csv") \
          .drop_duplicates("uid", keep="last")
rpp = pd.read_csv(f"{FPGO}/experiments_summary/data/fused/reinforce_full/per_problem.csv") \
        .drop_duplicates("uid", keep="last")
def line(df):
    n = len(df); c = int(df.correct.sum())
    by = {d: g.correct.mean() for d, g in df.groupby("dataset")}
    return c/n, n-c, by
sd_acc, sd_creg, sd_by = line(fused)
rp_acc, rp_creg, rp_by = line(rpp)
# combined-standalone rows (from RESULTS_FUSED.md canonical numbers)
with open(os.path.join(HERE, "fused_table.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
    f.write("Method & Acc & Cum.\\ regret & DS-1000 & DDXPlus & HotpotQA \\\\\n\\midrule\n")
    f.write("Base & 0.439 & 2365 & 0.318 & 0.436 & 0.521 \\\\\n")
    f.write("ICL $k{=}3$ & 0.533 & 1971 & 0.358 & 0.647 & 0.509 \\\\\n")
    f.write(f"REINFORCE++ (fused) & {rp_acc:.3f} & {rp_creg:.0f} & "
            f"{rp_by['ds1000']:.3f} & {rp_by['ddxplus']:.3f} & {rp_by['hotpotqa']:.3f} \\\\\n")
    f.write("SDFT standalone ($3\\times$) & 0.573 & 1800 & 0.423 & 0.664 & \\textbf{0.562} \\\\\n")
    f.write(f"\\textbf{{SDFT fused (ours)}} & \\textbf{{{sd_acc:.3f}}} & \\textbf{{{sd_creg:.0f}}} & "
            f"\\textbf{{{sd_by['ds1000']:.3f}}} & \\textbf{{{sd_by['ddxplus']:.3f}}} & {sd_by['hotpotqa']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ── Table 3: mastery (from eval CSVs) ──
on = pd.read_csv(f"{SB}/results/ds1000_eval_online/eval_per_problem.csv")
bt = pd.read_csv(f"{SB}/results/ds1000_eval_batch/eval_per_problem.csv")
b5 = pd.read_csv(f"{SB}/results/ds1000_eval_batch_n5/eval_per_problem.csv")
bs = pd.read_csv(f"{SB}/results/ds1000_base/per_problem.csv").drop_duplicates("problem_id", keep="last")
with open(os.path.join(HERE, "mastery_table.tex"), "w") as f:
    f.write("\\begin{tabular}{lrr}\n\\toprule\n")
    f.write("Model & Acc & Solved / 955 \\\\\n\\midrule\n")
    f.write(f"Base & {bs.correct.mean():.3f} & {int(bs.correct.sum())} \\\\\n")
    f.write(f"Batch SDFT ($n{{=}}1$, 4 epochs) & {bt.correct.mean():.3f} & {int(bt.correct.sum())} \\\\\n")
    f.write(f"Batch SDFT ($n{{=}}5$, 2 epochs) & {b5.correct.mean():.3f} & {int(b5.correct.sum())} \\\\\n")
    f.write(f"\\textbf{{Online SDFT (ours)}} & \\textbf{{{on.correct.mean():.3f}}} & \\textbf{{{int(on.correct.sum())}}} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("wrote bench_table.tex, fused_table.tex, mastery_table.tex")
