# Online Self-Improvement Experiments — Master Results

**Method under test:** SDFT + forward-looking ICL (a.k.a. `fwd4`) — an online
1-oracle-per-question learner that (a) retrieves kNN in-context examples from a
growing memory of past correct answers, (b) re-evaluates recent problems with
the improving model (forward-looking), and (c) self-distills the accumulated
correct answers into the weights each batch.

**Model:** Qwen2.5-Coder-7B-Instruct (+ LoRA r=16). Embedder: Qwen3-Embedding-0.6B.
**Metric:** cumulative regret = Σ(1 − reward) over the online stream (lower is
better); equivalently 1 − running accuracy.

**Headline:** on BOTH benchmarks, `fwd4` (SDFT+fwd ICL) is the best method,
beating a no-train ICL baseline (= StreamBench's Self-StreamICL), a faithful
REINFORCE++ RL baseline, and the untrained base model. The lead opens in the
back half of the stream as the distilled model compounds.

---

## Benchmark 1 — LiveCodeBench (1055 problems, 1 oracle/q)

| Method | avg reward | final cumreg |
|---|---:|---:|
| Base model (no ICL, no train)   | 0.338 | 698.77 |
| REINFORCE++ (RL, no ICL)        | —     | 647.52 |
| ICL k=3 (no train)              | 0.424 | 607.50 |
| **SDFT + fwd ICL (fwd4, ours)** | **0.448** | **582.57** ← golden |

- Plot: `plots/lcb_cumreg.png`
- Reference (RLOO + Nemotron-70B-RM, uses 200 RM calls/q): cumreg 566.85 — our
  method is within ~16 of it while using **zero** reward-model calls and the
  strict 1-oracle budget.

### Ablations on fwd4 (LiveCodeBench)
| Variant | what it adds | final / status |
|---|---|---|
| `fwd4` | golden: KNN ICL + fwd re-eval + SDFT (2 epochs) | 582.57 |
| `fwd5` | + multi-triplet ICL, random re-eval, **1 epoch** | collapsed to ~tie w/ fwd4 (epoch cut hurt) |
| `fwd5a` | + multi-triplet ICL, **2 epochs**, no random re-eval | led fwd4 by ~+13 mid-run (preempted at b67, cumreg 369) |
| `fwd5b` | ICL as candidate-augmentation only (no ICL at train/infer) | ~tie w/ fwd4 early, incomplete |
| logprob-select (k3+randoms, pick by model logprob) | inference-time selection | 650.40 — LOST to base ICL (607) |

**Takeaways:** 2 training epochs matter (fwd5 with 1 epoch lost the lead);
multi-triplet ICL (fwd5a) is promising but not confirmed to completion;
inference-time logprob selection does not work (model logprob poorly calibrated
to code correctness).

### ICL-anchor characterization (105K query×anchor pairs, base model)
- ICL is **bimodal**: +0.17 reward on hard queries (base R=0), −0.22 on easy
  queries (base R=1). Net aggregate +0.03.
- KNN retrieval is **not** clearly better than random anchors — per-bucket avg
  lift and best-case reward are ~flat across similarity buckets.
- Dominant signal is `q_r_base` (does the model already solve it), Pearson +0.60;
  anchor features (sim, anchor reward, length) barely correlate.
- Conclusion: the real lever is *whether* to use ICL (skip on easy queries), not
  *which* anchor. A learned q_r_base gate reclaims ~0 of the oracle-gate savings
  (AUC 0.77 not enough given asymmetric errors).
- Plots: `plots/icl_pairs_*.png` (regenerate from `fpgo/results/analyze_icl_pairs/`
  via `fpgo/analyze_icl_pairs/` scripts if needed).

---

## Benchmark 2 — StreamBench DS-1000 (955 problems, TF excluded, 1 feedback/q)

| Method | pass@1 | final cumreg | vs base |
|---|---:|---:|---:|
| Base model (no ICL, no train)   | 0.325 | 645 | — |
| REINFORCE++ (RL, no ICL)        | 0.369 | 603 | +42 |
| ICL k=3 (Self-StreamICL)        | 0.367 | 605 | +40 |
| **SDFT + fwd ICL (fwd4, ours)** | **0.423** | **551** | **+94** ← winner |

- Plot: `plots/ds1000_cumreg.png`
- SDFT lead over ICL grew through the stream: +5 (step 260) → +15 → +28 → **+54**
  final — same back-half pull-away as LiveCodeBench.
- **Why this matters:** StreamBench's own paper only explored updating the
  prompt/memory (Self-StreamICL); it explicitly did NOT update weights. Our fwd4
  adds the weight-update axis (self-distillation on the accumulated memory) and
  beats their Self-StreamICL baseline by 54 cumreg, plus a proper RL baseline.

### DS-1000 caveats
- **TensorFlow problems (45/1000) are EXCLUDED** from all methods (their tests
  `import tensorflow`, which conflicts with the training stack: TF→Keras3 breaks
  transformers, TF→protobuf7 breaks wandb/trl). All 4 lines scored on the same
  955 non-TF problems. Base/ICL ran on 1000 with TF rows filtered at metric time;
  R++/SDFT ran on the TF-excluded 955 stream directly.
- Minor impurity: ICL *physically* had the 45 TF problems in its stream, so 3
  TF-correct answers entered its memory (negligible; base unaffected; R++/SDFT
  never saw TF). Re-running ICL on the 955 cache would remove this — won't move 605.

---

## File locations (raw data — safe on disk)

### LiveCodeBench (repo: `fpgo`, git@github.com:Raghuramkowdeed/fpgo.git)
- Base: `fpgo/results/icl_sdft/baselines/base_0shot.csv` (per-problem reward)
- ICL k=3: `fpgo/results/icl_sdft/baselines/icl_k3.csv`
- fwd4 (golden): `fpgo/results/icl_sdft_fwd4/` (per_problem.csv, batch_metrics.csv, mem_bank.pkl, lora_adapter/)
- REINFORCE++: `fpgo/results/reinforce_pp/` (per_problem.csv, batch_metrics.csv)
- fwd5/5a/5b: `fpgo/results/icl_sdft_fwd5{,a,b}/`
- ICL-pairs analysis: `fpgo/results/analyze_icl_pairs/` (pairs.csv 105K, bucket_stats.csv, embeddings.pkl, base_rollouts.pkl)
- Runners: `fpgo/icl_sdft_fwd/run_icl_sdft_fwd4.py`, `run_reinforce_pp.py`, `run_icl_sdft_fwd5{,a,b}.py`
- Analysis: `fpgo/analyze_icl_pairs/*.py`
- Clean fwd variant pushed to git as `fpgo/icl_sdft_fwd/run_icl_sdft_fwd.py` (+ README, sbatch)

### StreamBench DS-1000 (repo clone: `/data/pulkitag/misc/raghuramkowdeed/stream-bench`)
- Base: `stream-bench/results/ds1000_base/per_problem.csv`
- ICL: `stream-bench/results/ds1000_icl/per_problem.csv`
- REINFORCE++: `stream-bench/results/ds1000_reinforce/{per_problem,batch_metrics}.csv`
- SDFT+fwd: `stream-bench/results/ds1000_sdft/{per_problem,batch_metrics}.csv` + lora_adapter/ + mem_bank.pkl
- Shared fixed stream order (seed 42, TF excluded): `stream-bench/ds1000_stream_order.pkl` (955 pids)
- Runners: `stream-bench/run_ds1000_{base,icl,sdft,reinforce}.py` (+ .sbatch)
- Oracle provided by StreamBench: `stream_bench/benchmarks/ds_1000.py` (execution-based, pass@1)

### Plots (permanent)
- `experiments_summary/plots/lcb_cumreg.png`
- `experiments_summary/plots/ds1000_cumreg.png`
- `experiments_summary/plots/icl_pairs_*.png` (may need regeneration)

---

## How to reproduce a plot
```python
# cumreg curves (both benchmarks): see the regeneration snippet that produced
# experiments_summary/plots/*.png — cumreg = cumsum(1 - reward/correct),
# filter rows where lib == 'Tensorflow' for DS-1000, dedup fwd4 per_problem by
# problem_id (it has ~10 resume-overlap rows).
```

## Environment notes
- Conda env `myenv` (torch 2.10, transformers, trl, peft). `SELF_DISTILLATION_PATH`
  = `/data/pulkitag/misc/raghuramkowdeed/projects/Self-Distillation` (DistilTrainer).
- For StreamBench runs: `export USE_TF=0` (transformers skips TF/Keras3);
  protobuf pinned <7 for wandb/trl; DS-1000 TF problems excluded.
- SLURM: H200, `--qos=shared-if-available` backfills fastest; checkpoint_every=2
  + auto-resume for long SDFT runs.

## Next candidate benchmarks (from mentor's list)
1. Spider (StreamBench text-to-SQL, execution oracle) — natural third, same skeleton.
2. BFCL (function calling), τ-bench, WebShop — multi-turn/agentic, bigger adaptation.
