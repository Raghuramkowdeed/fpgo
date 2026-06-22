#!/bin/bash
# Smoke run: 20 problems, batch_size 5, ~15 min on H200.
# Verifies the full pipeline (model load, ICL retrieval, oracle, SDFT step,
# checkpointing) end-to-end on a tiny slice.
#
#SBATCH --job-name=icl_sdft_smoke
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --account=csail
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.out
#SBATCH --error=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.err

set -euo pipefail

source /data/pulkitag/misc/raghuramkowdeed/setup_conda.sh myenv
export HF_HOME=/data/pulkitag/misc/raghuramkowdeed/hf_cache
export SELF_DISTILLATION_PATH=/data/pulkitag/misc/raghuramkowdeed/projects/Self-Distillation

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

python3 -u icl_sdft/run_icl_sdft_online1q.py \
    --output_dir results/icl_sdft_smoke \
    --batch_size 5 \
    --batch_window 5 \
    --max_problems 20 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --knn_k 3 \
    --icl_min_reward 0.8 \
    --reward_threshold 0.5 \
    --checkpoint_every 2

echo "DONE"
