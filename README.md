# fpgo

Frontier-Guided Policy Optimization (FGPO): RLOO training of a small code-LM
where a frontier LM proposes lightweight planning hints during training.
Held-out evaluation is hint-free.

## Layout

    fgpo/                          fgpo training code + sbatch wrappers
      run_fgpo_rloo.py             main training script (baseline + frontier modes)
      run_fgpo_rloo_baseline.sbatch    SLURM job: baseline (no hints)
      run_fgpo_rloo_frontier.sbatch    SLURM job: frontier-hint augmented
      frontier_client.py           Anthropic API wrapper for hint generation
      step1_smoke.py / .sbatch     standalone driver for the per-problem hint loop
      inspect_trained_gens.py      qualitative inspection of trained-policy outputs
      split_details.py             utilities for the train/test split
      fgpo_pseudocode.py           Python-style pseudocode of the algorithm
      FRONTIER_HINT_ANALYSIS.md    analysis of the frontier hint cache

    cumreg/                        shared package: dataset + oracle abstractions
                                   used by the fgpo scripts (Problem, CodeOracle, ...)

## Running

Baseline (no hints):

    sbatch fgpo/run_fgpo_rloo_baseline.sbatch

Frontier-hint augmented (requires ANTHROPIC_API_KEY in env):

    sbatch fgpo/run_fgpo_rloo_frontier.sbatch

Data files (problem statements + train/test split details) are NOT in this repo.
The scripts expect them at the paths set in the sbatch files.
