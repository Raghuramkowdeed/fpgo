# Forward-looking ICL+SDFT on LiveCodeBench (1 oracle / new q)

A drop-in variant of [`icl_sdft/`](../icl_sdft/) that adds a **window
re-evaluation** step to the per-batch loop. Same strict oracle budget on the
metric-cost side (one oracle per *new* question), same ICL retrieval, same
SDFT update — but the teacher example used in the SDFT step is now refreshed
under the *current* model + *current* in-context memory, rather than frozen
to whatever rollout was generated when the question first arrived.

In our LCB experiment this brings cumulative regret from `icl_sdft`'s 591.19
down to **~582** (with the same lr, schedule, and 1-oracle-per-new-q budget).

## Where things live

```
icl_sdft_fwd/
  README.md                       this file
  run_icl_sdft_fwd.py             the training/eval loop (~750 LOC)
  run_icl_sdft_fwd_smoke.sbatch   20-q smoke (~10-15 min)
  run_icl_sdft_fwd_full.sbatch    full LCB run (1055 q, ~7-9 h on H200)
  run_icl_sdft_fwd_resume.sbatch  resume from last checkpoint

../results/icl_sdft_fwd/
  per_problem.csv                 one row per NEW-batch q (greedy reward, cumreg)
  batch_metrics.csv               one row per outer batch (re-eval counts, timings)
  mem_bank_trajectory.csv         per-batch mem_bank stats (size, mean R, n_R≥0.8)
  forward_attempts.csv            per-row re-eval audit (pid, old_R, new_R, replaced)
  state.json                      resume state
  lora_adapter/                   final LoRA weights
  mem_bank.pkl, hist.pkl, pid_log.pkl, details.pkl
```

## What's different from `icl_sdft/`

The original `icl_sdft` loop, per batch:

  1. retrieve ICL → greedy gen on new batch → oracle (cumreg cost)
  2. push every new (q, code, r) to the cache (no reward gate on storage)
  3. SDFT one step over pairs with `r ≥ reward_threshold` from the last
     `1 + batch_window` batches, with the **rollout produced when each q first
     arrived** as the teacher demo

`icl_sdft_fwd` swaps step 3's teacher source with a **per-batch refresh**:

  1. same: ICL retrieve → greedy gen → oracle on NEW batch (cumreg cost)
  2. **NEW step**: for each pid in `pid_log[batch_idx - batch_window : batch_idx]`,
     retrieve ICL from the *current* `mem_bank` and re-run greedy gen + oracle
     **with the current model**. These oracle calls are training-only cost
     (they do **not** enter cumulative regret).
  3. **mem_bank update rule: max-R, tie → latest**. Re-eval that beats the
     existing entry replaces it. Re-eval that ties (most common at R = 1)
     still wins because of the tie-breaker — keeps the freshest rollout.
  4. SDFT pool = pids in (past_window ∪ current batch) whose
     `mem_bank[pid].reward ≥ reward_threshold`. Each pair uses the CURRENT
     mem_bank's ICL anchors and the CURRENT mem_bank's max-R rollout as the
     teacher demo. So pairs that the student has already partially absorbed
     (KL loss dropped) get retrained against an updated-but-still-best demo,
     and pairs whose mem_bank entry was just bumped get the new demo
     immediately.

A single data structure (`mem_bank`) is used for both ICL retrieval AND SDFT
teacher demo, with one invariant: `mem_bank[pid].response` is always the
max-reward rollout we've ever produced for that pid. A separate `hist[pid]`
keeps a chronological log of every (re-)attempt for offline analysis but is
not used in training.

## Cumulative-regret accounting

Same as `icl_sdft`:

```
cumreg = Σ over NEW-batch pids of (1 - greedy_reward)
```

Only the FIRST oracle call on a new q (step 1 above) contributes to cumreg.
The window re-eval oracle calls in step 2 are training-only cost, and the
mem_bank-updated rollout is what the SDFT teacher demonstrates against. This
preserves the "1 oracle per *new* question" claim on the metric while
allowing the training procedure to use more oracle calls per batch for
self-distillation purposes.

## Reproducing

Prerequisites (same as `icl_sdft`):

- H200 (or any ≥80 GB GPU). Holds student + teacher in bf16 + embedder.
- `Continual-Intelligence/Self-Distillation` cloned somewhere; point
  `SELF_DISTILLATION_PATH` at it (the sbatch files do this).
- LCB problem JSON at `<repo-root>/data/livecodebench_problems.json`.
- The `myenv` conda environment from the parent project.

Smoke (verifies end-to-end on a tiny slice):

```
sbatch icl_sdft_fwd/run_icl_sdft_fwd_smoke.sbatch
```

Full run:

```
sbatch icl_sdft_fwd/run_icl_sdft_fwd_full.sbatch
```

Resume if it hits walltime:

```
sbatch icl_sdft_fwd/run_icl_sdft_fwd_resume.sbatch
```

## Key hyperparameters

Defaults match `icl_sdft` exactly, so the comparison is apples-to-apples:

| Knob | Value | Notes |
|---|---:|---|
| `--model_name`        | `Qwen/Qwen2.5-Coder-7B-Instruct` |   |
| `--embedder_name`     | `Qwen/Qwen3-Embedding-0.6B`      |   |
| `--batch_size`        | 10  | problems per outer step (1 cumreg-cost oracle each) |
| `--batch_window`      | 9   | re-eval covers `pid_log[t-9:t]` → ≤90 past pids/batch |
| `--knn_k`             | 3   | ICL anchors retrieved per question |
| `--icl_min_reward`    | 0.8 | only past entries with R ≥ this qualify as anchors |
| `--reward_threshold`  | 0.5 | only mem_bank entries with R ≥ this enter SDFT |
| `--num_train_epochs`  | 2   | passes over the SDFT pool per outer step |
| `--learning_rate`     | 5e-5 |   |
| `--sdft_chunk_size`   | 50  | grad-accum chunk so big pools don't OOM |
| `--oracle_workers`    | 30  | multiprocessing.Pool for sandbox eval |
| LoRA r / alpha        | 16 / 32 | `all-linear` targets |

## What this is NOT

- **Not** best-of-N: still one greedy rollout per gen call.
- **Not** RL: the SDFT update is forward KL between teacher and student
  next-token distributions, not policy gradient.
- **Not** an extra reward model: oracle = sandboxed code execution against
  public test cases, the same oracle as `icl_sdft`.
- **Not** a memory expansion: same per-pid storage as the baseline (one
  rollout per pid, the max-R one).
