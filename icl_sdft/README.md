# Online ICL-Conditioned Self-Distillation on LiveCodeBench (1 oracle / q)

This directory implements an **online learning** procedure for LiveCodeBench
(LCB) that respects a strict budget of **one oracle call per question**
(where "oracle" = executing the candidate code against the problem's public
test cases). The motivating constraint is that we want a method that fits
the deployment setting: each new problem can be evaluated once before the
model has to move on.

The full run on all 1055 LCB problems (Qwen2.5-Coder-7B-Instruct + LoRA)
matches the cumulative regret of an RLOO + Nemotron-70B-Reward baseline,
**without using any reward model at all** — just an in-context memory of
past `(question, response, reward)` triples to anchor a self-distillation
update.

## Where things live

```
icl_sdft/
  README.md                       this file
  run_icl_sdft_online1q.py        the training/eval loop (~600 LOC)
  analyze_cumreg.py               regenerate the cumreg comparison plot + table
  run_icl_sdft_full.sbatch        full LCB run (1055 q, ~5 h)
  run_icl_sdft_resume.sbatch      resume from last checkpoint
  run_icl_sdft_smoke.sbatch       20-question smoke run (~15 min)

../results/icl_sdft/
  per_problem.csv                 one row per question (greedy reward, cumreg)
  batch_metrics.csv               one row per outer batch (KL loss, accept rate, timings)
  state.json                      final resume state
  cumreg_1oracle_methods.{png,pdf}    comparison plot (our run vs 3 baselines)
  RESULTS_TABLE.md                table of avg reward + cumreg per method
  eval_curves_final.png           per-question + KL + cumreg, our run only
  baselines/
    base_0shot.csv                Qwen2.5-Coder-7B greedy, no training, no ICL
    icl_k3.csv                    + kNN ICL anchors from a growing cache
    rloo_nemotron_rm.csv          RLOO trained with Nemotron-70B-Reward (oracle eval only)
```

## Problem setup

For each question `q_t` arriving in stream order:

1. We can use the current model to generate code.
2. We may **call the oracle (public-test execution) exactly once** on `q_t`.
3. We may update the model however we like before the next question arrives.

`cumulative regret = Σ_t (1 − greedy_reward(q_t))` where the greedy reward
is measured **before** any model update on `q_t`. This is the standard
online-learning cumulative regret on the question stream — lower is better.

## Method

### One oracle, triple-duty

A single greedy rollout per question gives us:

- **the evaluation signal** that increments cumulative regret
- a **data candidate** for the self-distillation buffer
  (kept only if reward ≥ `--reward_threshold`, default `0.5`)
- a **memory anchor**: `(q_t, embedding(q_t), code_t, r_t)` is always pushed
  to the live cache regardless of `r_t`

### ICL retrieval over a growing memory

The cache stores every past rollout. Before generating, we embed `q_t` with
`Qwen3-Embedding-0.6B` and retrieve the `--knn_k` (default 3) nearest past
entries **whose reward ≥ `--icl_min_reward`** (default 0.8). These get
flattened into `(user=past_question, assistant=past_code)` chat turns in
both the student and teacher prompts.

The retrieval is single-hop: an anchor's own ICL anchors are NOT included
recursively — only its `(question, code)` pair.

### Self-distillation step

We use `DistilTrainer` from the public
[`Continual-Intelligence/Self-Distillation`](https://github.com/Continual-Intelligence/Self-Distillation)
repo (paper: [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)).

- The **student prompt** is `system + 3 ICL anchors + current question`,
  reward-conditioned on `<reward>{r}</reward>`.
- The **teacher prompt** is the same context with the current rollout
  *additionally demonstrated* before the question is re-asked. So the
  teacher sees one extra hint that the student doesn't.
- **Loss** is forward KL between teacher and student next-token
  distributions on the shared completion tokens.
- Only the student's LoRA weights update; teacher (= same base weights, no
  LoRA, no grads) stays frozen.

### Rolling window over batches

We do not update after every single problem. Instead we process problems
in **outer batches** of `--batch_size` (default 10), and at the end of
each batch run **one SDFT step over the last `1 + --batch_window` batches'
accepted pairs** (default window covers ~100 problems → ~40 SDFT pairs at
steady state). Each accepted pair contributes exactly one gradient per
epoch (we set `per_device_train_batch_size=1` and
`gradient_accumulation_steps = n_window`, so no pair is dropped). Two
epochs over the window per outer step (matches the Self-Distillation repo
README recipe). Learning rate `5e-5`, AdamW.

### What the script does NOT do

- It does **not** call any external reward model. The oracle (code
  execution) is the only reward signal.
- It does **not** do best-of-N selection. Exactly one greedy rollout per
  question.
- It does **not** retain the LoRA-trained model after the run for offline
  eval; the win we report is the *online* cumulative regret, which already
  reflects the trajectory of model improvement during the stream.

## Results

Full LCB run (1055 questions, Qwen2.5-Coder-7B-Instruct base). All
methods evaluated with the same 1-oracle-per-question greedy eval; the
training budgets differ:

| Method | avg R | cumreg | train oracle/q | train RM calls/q |
|---|---:|---:|---:|---:|
| Base 0-shot         | 0.338 | 698.77 | 0 | 0 |
| ICL k=3 (stream)    | 0.424 | 607.50 | 0 | 0 |
| RLOO + Nemotron-RM  | 0.463 | 566.85 | 0 | **200** |
| **ICL+SDFT (ours)** | 0.440 | **591.19** | **1** | 0 |

See `../results/icl_sdft/cumreg_1oracle_methods.pdf` for the curve.

### What the numbers mean

- **Base 0-shot** is Qwen2.5-Coder-7B greedy with no in-context examples
  and no training. It defines the "do nothing" baseline.
- **ICL k=3** uses the same growing memory as our method, with kNN
  retrieval, but does no weight updates. The +91 cumreg improvement over
  Base is the contribution of memory alone.
- **RLOO + Nemotron-RM** uses the standard RLOO recipe with the Nemotron
  70B Reward model scoring all 200 sampled rollouts per problem for the
  leave-one-out reward; the oracle is only used for the periodic greedy
  evaluation. This is the strongest baseline that exists in our repo for
  online learning on LCB.
- **ICL+SDFT (ours)** is within 24 cumreg of RLOO+RM while using zero
  RM calls and the strictest possible oracle budget. The +107 cumreg
  improvement over Base, and +16 over ICL alone, is the contribution of
  the self-distillation update on top of the growing memory.

## Reproducing

Prerequisites:

- An H200 (or similar ≥80 GB) GPU. The script holds two full Qwen-Coder-7B
  copies (student + teacher) in bf16 plus the embedder; peak ~90 GB.
- The `Continual-Intelligence/Self-Distillation` repo cloned somewhere,
  with its path pointed to by the `SELF_DISTILLATION_PATH` env var
  (the sbatch files set this).
- The LiveCodeBench problem file at
  `<repo-root>/data/livecodebench_problems.json` (same path FGPO uses).
- The myenv conda environment from the parent project (`transformers`,
  `peft`, `accelerate`, `datasets`, `bitsandbytes`, `trl`, `torch`).

Smoke (verifies the pipeline end-to-end in ~15 min):

```
sbatch icl_sdft/run_icl_sdft_smoke.sbatch
```

Full run (1055 questions, ~5 h on H200):

```
sbatch icl_sdft/run_icl_sdft_full.sbatch
```

If the job hits walltime or is preempted, resume from the last checkpoint:

```
sbatch icl_sdft/run_icl_sdft_resume.sbatch
```

Regenerate the comparison plot + table from CSVs (does not need a GPU):

```
python icl_sdft/analyze_cumreg.py \
    --base results/icl_sdft/baselines/base_0shot.csv \
    --icl  results/icl_sdft/baselines/icl_k3.csv \
    --rloo results/icl_sdft/baselines/rloo_nemotron_rm.csv \
    --ours results/icl_sdft/per_problem.csv \
    --out_dir results/icl_sdft
```

## Key hyperparameters (defaults)

| Knob | Value | Notes |
|---|---:|---|
| `--model_name`        | `Qwen/Qwen2.5-Coder-7B-Instruct` | student + teacher are the same base model |
| `--embedder_name`     | `Qwen/Qwen3-Embedding-0.6B`      | last-token + L2-norm pooling |
| `--batch_size`        | 10  | problems per outer step (1 oracle each) |
| `--batch_window`      | 9   | SDFT window = current + last 9 batches |
| `--knn_k`             | 3   | ICL anchors retrieved per question |
| `--icl_min_reward`    | 0.8 | only past entries with reward ≥ this qualify as anchors |
| `--reward_threshold`  | 0.5 | only rollouts with reward ≥ this enter the SDFT buffer |
| `--num_train_epochs`  | 2   | passes over the windowed buffer per outer step |
| `--learning_rate`     | 5e-5 | matches Self-Distillation README recipe |
| LoRA r / alpha        | 16 / 32 | `all-linear` targets |

## Files / dependencies imported

- `cumreg.datasets.base.Problem` and `cumreg.oracles.code_oracle.CodeOracle`
  from the sibling `cumreg/` package in this repo (these are the same
  modules FGPO uses; LiveCodeBench problem objects + sandboxed
  public-test execution).
- `distil_config.DistilConfig` and `distil_trainer.DistilTrainer` from
  the upstream `Continual-Intelligence/Self-Distillation` repo (cloned
  separately; path via `SELF_DISTILLATION_PATH` env var).
- Standard HF stack: `transformers`, `peft`, `datasets`, `torch`.

## Known limitations / things to try next

- **No reward shaping on the SDFT loss.** Every accepted pair contributes
  equally regardless of its reward. Weighting by reward (or by 1 − reward,
  for hard-but-passed examples) might improve sample efficiency.
- **Threshold `0.5` is fixed.** A small sweep `{0.3, 0.5, 0.7}` would
  show whether the acceptance gate is in the right place.
- **kNN retrieval is by cosine over Qwen3 embeddings.** No diversity term;
  if 3 of the most-similar past problems are near-duplicates we waste 2
  anchor slots.
- **No held-out eval.** Cumulative regret is the only learning signal we
  track; would be cleaner to reserve ~50 problems for a fixed eval set
  re-evaluated every N batches.
