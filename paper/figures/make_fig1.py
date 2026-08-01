#!/usr/bin/env python3
"""Figure 1 (headline results, vector PDF). Panel A: streaming accuracy on the 7
StreamBench tasks, ours vs strongest baseline per task. Panel B: rolling accuracy
on the fused 3-domain stream, ours vs REINFORCE++ (collapse). Data: committed CSVs
+ the verified benchmark table (same numbers as tables/make_tables.py)."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FPGO = os.path.abspath(os.path.join(HERE, "..", ".."))

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a897f"

ROWS = [  # task, base, icl, rpp, ours
    ("LCB", .338, .424, .386, .448),
    ("DS-1000", .325, .367, .369, .423),
    ("Spider", .747, .778, .587, .808),
    ("DDXPlus", .436, .647, .404, .664),
    ("BIRD", .330, .346, .271, .354),
    ("HotpotQA", .521, .509, .537, .562),
    ("ToolBench", .549, .593, .607, .615),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 2.9), dpi=200,
                               gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor("white")

# Panel A: ours vs best baseline
tasks = [r[0] for r in ROWS]
best_bl = [max(r[1], r[2], r[3]) for r in ROWS]
ours = [r[4] for r in ROWS]
x = np.arange(len(tasks)); w = 0.38
ax1.bar(x - w/2, best_bl, w, color=GRAY, label="best baseline", zorder=3)
ax1.bar(x + w/2, ours, w, color=BLUE, label="​Online SDFT (ours)", zorder=3)
for i, (b, o) in enumerate(zip(best_bl, ours)):
    ax1.annotate(f"+{(o-b)*100:.1f}", (x[i] + w/2, o), xytext=(0, 2),
                 textcoords="offset points", ha="center", fontsize=6.5, color=BLUE)
ax1.set_xticks(x); ax1.set_xticklabels(tasks, fontsize=7.5, rotation=20)
ax1.set_ylabel("streaming accuracy", fontsize=8)
ax1.set_ylim(0, 0.9)
ax1.legend(fontsize=7, frameon=False, loc="upper left")
ax1.set_title("Best on all seven StreamBench tasks", fontsize=9, loc="left")
ax1.grid(axis="y", color="#eceae4", lw=0.6, zorder=0)
for sp in ["top", "right"]: ax1.spines[sp].set_visible(False)
ax1.tick_params(labelsize=7.5)

# Panel B: fused stream rolling accuracy, ours vs RPP
sd = pd.read_csv(f"{FPGO}/experiments_summary/data/fused/sdft/per_problem.csv") \
       .drop_duplicates("uid", keep="last").sort_values("step").reset_index(drop=True)
rp = pd.read_csv(f"{FPGO}/experiments_summary/data/fused/reinforce_full/per_problem.csv") \
       .drop_duplicates("uid", keep="last").sort_values("step").reset_index(drop=True)
W = 300
ax2.plot(sd.index, sd.correct.rolling(W, min_periods=100).mean(), color=BLUE,
         lw=1.8, label="​Online SDFT (ours)")
ax2.plot(rp.index, rp.correct.rolling(W, min_periods=100).mean(), color=ORANGE,
         lw=1.8, label="REINFORCE++")
ax2.axhline(0.439, color=GRAY, lw=1.0, ls=":", label="zero-shot")
ax2.set_xlabel("stream position (3-domain fused stream)", fontsize=8)
ax2.set_ylabel("rolling accuracy", fontsize=8)
ax2.set_ylim(-0.02, 0.75)
ax2.legend(fontsize=7, frameon=False, loc="center left")
ax2.set_title("Stable under distribution shift; RL collapses", fontsize=9, loc="left")
ax2.grid(axis="y", color="#eceae4", lw=0.6)
for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
ax2.tick_params(labelsize=7.5)

fig.tight_layout(w_pad=2.0)
fig.savefig(os.path.join(HERE, "fig1.pdf"), bbox_inches="tight")
print("wrote fig1.pdf")
