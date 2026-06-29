#!/bin/bash
# Full LCB run: 1055 problems, bs=10, batch_window=9 (rolling 100-problem
# window for both window re-eval and SDFT), strict 1 oracle/new-q.
# Re-eval oracle calls are training-only cost (not counted in cumreg).
# Expect ~7-9 h wall on H200.
#
#SBATCH --job-name=icl_sdft_fwd_full
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --account=csail
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --output=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.out
#SBATCH --error=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.err

set -euo pipefail

source /data/pulkitag/misc/raghuramkowdeed/setup_conda.sh myenv
export HF_HOME=/data/pulkitag/misc/raghuramkowdeed/hf_cache

# Self-Distillation repo (https://github.com/Continual-Intelligence/Self-Distillation)
export SELF_DISTILLATION_PATH=/data/pulkitag/misc/raghuramkowdeed/projects/Self-Distillation

cd /data/pulkitag/misc/raghuramkowdeed/projects/fpgo

python3 -u icl_sdft_fwd/run_icl_sdft_fwd.py \
    --output_dir results/icl_sdft_fwd \
    --batch_size 10 \
    --batch_window 9 \
    --num_train_epochs 2 \
    --learning_rate 5e-5 \
    --knn_k 3 \
    --icl_min_reward 0.8 \
    --reward_threshold 0.5 \
    --oracle_workers 30 \
    --sdft_chunk_size 50 \
    --checkpoint_every 5

echo "DONE"
