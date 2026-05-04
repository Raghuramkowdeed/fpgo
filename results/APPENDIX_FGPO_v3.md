# Appendix: Self-Coached FGPO (v3) on LiveCodeBench

## Method

We train a small policy (Qwen2.5-Coder-7B-Instruct + LoRA) with RLOO on
LiveCodeBench coding problems. Each training batch is *augmented*: every
eligible problem contributes two rows to the RLOO update — one whose
prompt is the bare problem statement, and one whose prompt is the
problem statement prepended with a short *hint*. The hint is meant to
nudge the policy toward useful reasoning patterns during rollout.

In **v3**, the hint generator is the **same RL policy that is being
trained** (the LoRA-adapted Qwen, not an external API model). Concretely,
for each eligible problem, before the augmented training row is built we
run a short *Step-1 loop* of up to 3 rounds. In each round:

1. The current policy samples 10 candidate solutions (`probe_samples=10`,
   `T=1.0`) for the problem.
2. The same policy is then re-prompted with a "coach" system message
   asking it to produce a short JSON-formatted hint, conditioned only on
   the problem statement and the **raw 10 code samples** it just
   produced. **No reward signal, no oracle feedback, and no
   best/worst labels** are exposed to the hint-generation step. The
   model must infer likely failure modes from its own raw code.
3. The hint is cached and used as the prefix for the next probe round.

The best hint across rounds (judged by oracle reward only for caching;
the reward is *not* shown to the hint generator) is stored. During
training, the augmented row uses this cached hint. The held-out test
evaluation is **hint-free** — greedy decoding on the bare problem
statement.

The hint-generation prompt is intentionally written with strong
anti-code wording ("you are a coach, not a coder; output JSON only; no
Python code") to counteract the LoRA's code-generation bias. We use
greedy decoding for the hint with `max_new_tokens=300` and a 32k input
context window to comfortably hold the problem plus 10 raw code samples.

## Setup

| Component | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Adapter | LoRA, r=16, α=32, target_modules=all-linear, trainable params 40.4M (0.53%) |
| Dataset | LiveCodeBench, 600 train / 455 test (we use 100 of 455 for per-batch test eval and 400 for the final eval) |
| RL algorithm | RLOO (TRL implementation), KL coef β=0.01, lr=5e-5, num_train_epochs=1 |
| Schedule | 12 outer batches × `batch_size=50` problems, `num_generations=10` rollouts per row, `per_device_train_batch_size=10`, `gradient_accumulation_steps=10` (eff 100 samples / optim step) |
| Decoding | Rollouts and Step-1 probes at `T=1.0`; held-out greedy eval is `T=0`, `do_sample=False` |
| Hint eligibility | First 50% of training problems (300 / 600) |
| Hint loop (v3) | 3 rounds, 10 probe samples per round, 32k input context, 300 max output tokens, greedy, JSON-formatted output |
| Hardware | 1× NVIDIA H200 (140 GB), 256 GB host RAM, 16 CPUs |

## Results

### Final greedy accuracy (400 held-out problems)

| Variant | Final greedy | Δ vs. baseline |
|---|---:|---:|
| **RLOO baseline (no hints)** | **0.5257** | — |
| **FGPO v3 (Qwen self-hint, no rewards)** | **0.5434** | **+0.018** |

For context, two ablations of FGPO with an external Claude frontier
model as the hint generator:

| Variant | Final greedy | Δ vs. baseline |
|---|---:|---:|
| FGPO v1 — Claude sees rewards + oracle errors | 0.5192 | −0.007 |
| FGPO v2 — Claude sees raw samples only (no rewards) | 0.5494 | +0.024 |

### Per-batch held-out test accuracy (greedy on first 100 test problems)

| Batch | Baseline | FGPO v3 |
|---:|---:|---:|
| 0 | 0.3125 | 0.3142 |
| 1 | 0.2983 | 0.4242 |
| 2 | 0.3067 | 0.4792 |
| 3 | (skipped) | 0.5308 |
| 4 | 0.4133 | 0.5458 |
| 5 | 0.4375 | 0.5417 |
| 6 | 0.4633 | 0.5608 |
| 7 | 0.5050 | 0.5225 |
| 8 | (skipped) | 0.5208 |
| 9 | 0.5300 | 0.5300 |
| 10 | 0.5433 | 0.5158 |
| 11 | 0.5550 | 0.5042 |
| **Final 400** | **0.5257** | **0.5434** |

v3 reaches 0.50+ by batch 4 — about three batches faster than the
baseline (which reaches 0.50 only at batch 7). The two trajectories
converge in the back half (batches 8–11), with the baseline catching up
on its larger pool of bare-prompt rollouts.

### Hint impact during training (Step-1 cache statistics, 300 problems)

This shows what the hints did to *training-time* rollout reward —
**not** the trained policy's test accuracy.

| Statistic | Value |
|---|---:|
| Baseline mean reward (round 0, no hint) | 0.342 |
| Hinted mean reward (best of 3 hint rounds) | 0.353 |
| Δ mean (hinted − baseline) | **+0.012** |
| Helped (Δ > +0.05) | 24.0% |
| Hurt (Δ < −0.05) | 16.3% |
| Neutral (\|Δ\| ≤ 0.05) | 59.7% |

The Qwen self-coach almost never improves its own training-time
pass-rate (Δ ≈ +0.012 on average; 60% of hints are inert; 16% slightly
hurt). Yet the trained policy's held-out greedy accuracy rises by
+1.8 percentage points over the baseline.

### Interpretation

The Step-1 cache statistics suggest the hint *content* is not
load-bearing in v3 — the policy cannot meaningfully coach itself. What
remains useful is the **augmented-row mechanism**: every eligible
problem contributes two RLOO groups instead of one (a no-hint group and
a hint-prefixed group), each with its own leave-one-out advantage. The
extra rollouts and the prompt diversity (with vs without a generic
preamble) appear to be the active ingredients, not the specific
reasoning conveyed by the hint.

This matches the v1 vs v2 contrast: removing the reward feedback from
the frontier prompt actually *improves* the trained policy (v1 → v2 is
+0.030), suggesting that very-informative reward-aware hints push the
model toward an overfit "hint-conditioned" solution distribution that
does not transfer to hint-free greedy decoding.

The practical takeaway is that FGPO can be deployed without any
external model and without any oracle calls during the hint loop,
recovering most of the training-time benefit at a fraction of the
infrastructure cost.

## Files

- Plot — final greedy bar chart for all four variants:
  `plot_fgpo_final_greedy_bar.png`
- Plot — per-batch test trajectory, v3 vs baseline:
  `plot_fgpo_v3_vs_baseline.png`
- Plot — per-batch test trajectory, all four variants:
  `plot_fgpo_v1_v2_v3_vs_baseline.png`
- Raw per-batch CSV (each variant): `results_fgpo_rloo_*/test_eval.csv`
- Raw final eval JSON (each variant): `results_fgpo_rloo_*/final_eval.json`
- v3 hint cache (300 problems, full per-round records):
  `cache/hints_frontier_v3.pkl`
- v3 entry point script: `run_fgpo_rloo_v3.py`
- v3 hint-generation client: `local_hint_client.py`
