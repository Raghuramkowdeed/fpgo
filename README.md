# fpgo

Frontier-Guided Policy Optimization (FGPO): RLOO training of a small code-LM
where a frontier LM proposes lightweight planning hints during training.
Held-out evaluation is hint-free.

## Layout

    fgpo/                  FGPO training code + sbatch wrappers
    cumreg/                shared package: Problem, oracle, dataset abstractions

## `fgpo/` — components

### Training entrypoint
- **`run_fgpo_rloo.py`** — main training script. Implements the outer
  loop: per-batch greedy pre-eval, optional Step-1 frontier hint
  construction, RLOO update on the (no-hint + hinted) augmented batch,
  greedy post-eval, periodic test-set eval. Writes `training_log.csv`,
  `test_eval.csv`, LoRA checkpoints, optimizer/scheduler state, and
  resumable `state.json`. Same script is used for the baseline run
  (just omit `--use_frontier_hints`).

### Frontier hint loop
- **`frontier_client.py`** — Anthropic Claude wrapper. Given a problem
  + a history of (hint tried, sample codes, rewards, oracle errors),
  asks Claude to propose the next hint. Reads `ANTHROPIC_API_KEY` from
  env. Used inside `run_step1_loop` in the main script.
- **`step1_smoke.py`** — standalone driver for the Step-1 frontier
  loop with no training attached. Useful for debugging the hint
  pipeline, building the hint cache offline, and inspecting per-round
  results.

### SLURM submission
- **`run_fgpo_rloo_baseline.sbatch`** — H200, no hints, RLOO only.
- **`run_fgpo_rloo_frontier.sbatch`** — H200, frontier-hint augmented
  (`--use_frontier_hints --frontier_fraction 0.5`). Requires
  `ANTHROPIC_API_KEY` in env.
- **`step1_smoke.sbatch`** — wrapper for `step1_smoke.py`.
- **`import_test.sbatch`** — sanity-check job that just imports the
  modules (catches dependency / path issues quickly).

### Inspection / utilities
- **`inspect_trained_gens.py`** — loads a trained LoRA + base model,
  generates greedy completions on a sample of test problems with both,
  saves raw outputs side-by-side for manual qualitative comparison.
- **`split_details.py`** — one-time script that splits the upstream
  `details.pkl` into `details_train.pkl` (first 600) +
  `details_test.pkl` (rest).
- **`fgpo_pseudocode.py`** — Python-style pseudocode of Algorithms 1
  (Step-1 hint loop) and 2 (FGPO-RLOO outer loop). Reference, not
  runnable.
- **`FRONTIER_HINT_ANALYSIS.md`** — analysis of the cached frontier
  hints: aggregate effect, multi-round refinement stats, what
  patterns help vs. hurt.

## `cumreg/` — shared dependencies used by FGPO

Only `Problem` and `CodeOracle` are imported by FGPO; the other
modules are part of the parent ICL/cumulative-regret framework and
are kept for completeness so the package imports cleanly.

### Used by FGPO
- **`datasets/base.py`** — defines `Problem` (dataclass: id, question,
  public/private tests, etc.) and `DatasetStreamer` (abstract iterator
  over problems).
- **`datasets/livecodebench.py`** — `LiveCodeBenchStreamer`: parses
  the LiveCodeBench JSON dump into `Problem` objects.
- **`oracles/base.py`** — `Oracle` abstract base (evaluate a response
  against a problem; return reward + optional feedback string).
- **`oracles/code_oracle.py`** — `CodeOracle`: extracts Python code
  from a model response, executes it against the problem's hidden
  tests in a subprocess, returns fractional pass-rate as reward and a
  trimmed traceback as feedback.

### Not directly used by FGPO (kept so the package imports)
- **`__init__.py`** — re-exports the public API.
- **`config.py`** — `Config` dataclass for ICL/cumreg experiments.
- **`agent.py`** — Agent-Lightning `@rollout` agent.
- **`algorithm.py`** — sequential-task ICL algorithm with regret
  tracking.
- **`embedder.py`** — extracts hidden states from the generation
  model, mean-pools over non-pad tokens (used as ICL example
  embeddings).
- **`engine.py`** — batched HF `generate()` engine with best-of-N
  support.
- **`experiment.py`** — CSV logging, resume, regret bookkeeping.
- **`formatting.py`** — chat-template-aware prompt builder for
  single-turn / multi-turn ICL with optional reasoning traces.
- **`retriever.py`** — kNN / recency / diversity-based ICL example
  retrieval.
- **`datasets/gsm8k.py`** — GSM8K streamer (math).
- **`oracles/math_oracle.py`** — math oracle for GSM8K-style problems.

## Running

Baseline (no hints):

    sbatch fgpo/run_fgpo_rloo_baseline.sbatch

Frontier-hint augmented (requires `ANTHROPIC_API_KEY`):

    sbatch fgpo/run_fgpo_rloo_frontier.sbatch

Data files (LiveCodeBench problems, train/test split details, hint
cache) are NOT in this repo. The scripts expect them at the absolute
paths set in the sbatch files.
