#!/bin/bash
#SBATCH --job-name=fgpo_step1_smoke
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=lab-free
#SBATCH --account=csail
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.out
#SBATCH --error=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.err

set -euo pipefail

source /data/pulkitag/misc/raghuramkowdeed/setup_conda.sh myenv

# ANTHROPIC_API_KEY must be in ~/.bashrc; source it for non-login shell safety.
source ~/.bashrc

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set" >&2
    exit 1
fi

cd /data/pulkitag/misc/raghuramkowdeed/projects/agl

python -u fgpo/step1_smoke.py \
    --output_dir /data/pulkitag/misc/raghuramkowdeed/projects/agl/fgpo/results_step1_smoke \
    --n_problems 50 \
    --n_samples 10 \
    --n_rounds 5 \
    --seed 42 \
    --max_new_tokens 1024 \
    --gen_batch_size 50 \
    --frontier_model claude-sonnet-4-6 \
    --frontier_workers 8 \
    --worst_k 2 \
    --best_k 2 \
    --oracle_timeout 60
