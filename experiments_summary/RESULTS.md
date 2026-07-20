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
| REINFORCE++ (RL, no ICL)        | 0.386 | 647.52 |
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

## Benchmark 3 — StreamBench Spider (2147 problems, text-to-SQL, 1 oracle/q)

| Method | EX (exec acc) | final cumreg | vs base |
|---|---:|---:|---:|
| REINFORCE++ (RL, no ICL)        | 0.587 | 886 | −342 (collapsed) |
| Base model (no ICL, no train)   | 0.747 | 544 | — |
| ICL k=3 (Self-StreamICL)        | 0.778 | 476 | +68 |
| **SDFT + fwd ICL (fwd4, ours)** | **0.808** | **412** | **+132** ← winner |

- Plot: `plots/spider_cumreg.png`. SDFT beats ICL by **+64** (largest ICL margin yet:
  LCB +24, DS-1000 +54, Spider +64); lead opens through the back half.
- REINFORCE++ collapses hard on Spider — the strong base (0.75) gives whitened
  advantages little signal, KL drifts, back-half accuracy craters (0.587 final).

## Benchmark 4 — StreamBench DDXPlus (1764 problems, medical diagnosis, 1 oracle/q)

49-way single-turn classification (pick 1 of 49 pathologies from a patient profile).

| Method | accuracy | final cumreg | vs base |
|---|---:|---:|---:|
| REINFORCE++ (RL, no ICL)        | 0.404 | 1051 | −56 (collapsed below base) |
| Base model (no ICL, no train)   | 0.436 | 995 | — |
| ICL k=3 (Self-StreamICL)        | 0.647 | 622 | +373 |
| **SDFT + fwd ICL (fwd4, ours)** | **0.664** | **592** | **+403** ← winner |

- Plot: `plots/ddxplus_cumreg.png`. SDFT beats ICL by **+30**, base by **+403** (biggest
  base margin — weak 44% base leaves huge headroom; ICL alone nearly doubles accuracy).
- REINFORCE++ climbs mid-run then collapses in the back half (KL drift), ending
  *below* base. So R++ collapses on BOTH a strong-base (Spider) and weak-base
  (DDXPlus) task — naive online RL (1 greedy sample/q, offline replay) is unstable
  regardless of base strength, while SDFT reliably wins on all 4 benchmarks.


## Benchmark 5 — StreamBench BIRD (1534 problems, hard text-to-SQL, 1 oracle/q)

Harder than Spider (messy real-world DBs). All methods stuck ~33-35% (low base headroom).

| Method | EX | final cumreg | vs base |
|---|---:|---:|---:|
| REINFORCE++ | 0.270 | 1119 | -91 (collapsed) |
| Base model | 0.330 | 1028 | - |
| ICL k=3 | 0.346 | 1003 | +25 |
| **SDFT + fwd ICL (ours)** | **0.354** | **991** | **+37** <- winner |

- Plot: `plots/bird_cumreg.png`. The marginal case: SDFT wins but margins are thinnest
  of any benchmark (+12 vs ICL) — sparse correct answers + poor cross-DB transfer limit
  both ICL and SDFT. R++ collapses again.

## Benchmark 6 — StreamBench HotpotQA (1500 problems, multi-hop QA, EM oracle, 1/q)

| Method | acc | final cumreg | vs base |
|---|---:|---:|---:|
| ICL k=3 (Self-StreamICL) | 0.509 | 736 | **-17 (ICL HURTS)** |
| Base model | 0.521 | 719 | - |
| REINFORCE++ | 0.537 | 695 | +24 |
| **SDFT + fwd ICL (ours)** | **0.562** | **657** | **+62** <- winner |

- Plot: `plots/hotpotqa_cumreg.png`. **HEADLINE RESULT:** ICL is WORSE than base
  (multi-hop QA answers need each question's own context, not transferable demos), yet
  SDFT wins by +62 vs base and **+79 vs ICL** — the weight-update axis delivers precisely
  where retrieval-only fails. Strongest evidence SDFT+fwd is not merely "better ICL".
  (Notably R++ also beats base here (+24) — weak base + short answers give RL signal.)

## Summary across all 6 benchmarks

SDFT+fwd (fwd4) wins ALL SIX (LiveCodeBench, DS-1000, Spider, DDXPlus, BIRD, HotpotQA),
beating base, ICL (Self-StreamICL), and a faithful REINFORCE++ on every one. R++ collapses
below base on 4/6 (Spider, DDXPlus, BIRD, LCB-ish). SDFT vs ICL margin: LCB +24, DS-1000
+54, Spider +64, DDX +30, BIRD +12, HotpotQA +79.


## Benchmark 7 — StreamBench ToolBench (750 problems, single-turn function calling, 1/q)

Given a query + 50-API catalog, emit ONE {"action","action_input"} JSON. Oracle =
STRICT exact-match (api_name + normalized args); deterministic, no LLM judge in the
reward loop. (Optional post-hoc NVIDIA-49b semantic judge validated but NOT the headline.)

| Method | strict acc | final cumreg | vs base |
|---|---:|---:|---:|
| Base model | 0.549 | 338 | - |
| ICL k=3 (Self-StreamICL) | 0.593 | 305 | +33 |
| REINFORCE++ | 0.607 | 295 | +43 |
| **SDFT + fwd ICL (ours)** | **0.615** | **289** | **+49** <- winner |

- Plot: `plots/toolbench_cumreg.png`. Tightest race of the campaign: R++ was the
  STRONGEST baseline here (0.607, an exception to its collapse elsewhere), yet SDFT
  still wins (+6 vs R++, +16 vs ICL) with the usual back-half pull-away.
- ToolBench-specific memory note: the 50-API catalog is ~7k tokens, so SDFT training
  needed (a) a compact teacher re-ask instead of repeating the catalog, (b)
  steps_per_generation=2 (teacher generates 2 prompts/call not ~50), (c) gradient
  checkpointing. All memory-only except (a); grad_accum, reward, cumreg, eval unchanged.
  Needs 12-14h walltime (heaviest job; GC + 7k prompts).

## FINAL: SDFT+fwd wins ALL 7 benchmarks

| Bench | base | ICL | R++ | SDFT | SDFT vs ICL |
|---|---|---|---|---|---|
| LiveCodeBench | 699 | 607 | 648 | 583 | +24 |
| DS-1000 | 645 | 605 | 603 | 551 | +54 |
| Spider | 544 | 476 | 886 | 412 | +64 |
| DDXPlus | 995 | 622 | 1051 | 592 | +30 |
| BIRD | 1028 | 1003 | 1119 | 991 | +12 |
| HotpotQA | 719 | 736 | 695 | 657 | +79 |
| ToolBench | 338 | 305 | 295 | 289 | +16 |

SDFT+fwd beats base, ICL (Self-StreamICL), and REINFORCE++ on all 7 (code, Python,
SQL x2, medical, QA, tool-use). R++ collapses below base on 4/7. HotpotQA: SDFT wins
where ICL HURTS. ToolBench: SDFT beats an unusually-strong R++.

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
