#!/bin/bash
#SBATCH --job-name=fused_rpp
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --account=csail
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.out
#SBATCH --error=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.err
set -euo pipefail
source /data/pulkitag/misc/raghuramkowdeed/setup_conda.sh myenv
export HF_HOME=/data/pulkitag/misc/raghuramkowdeed/hf_cache
export USE_TF=0
export SELF_DISTILLATION_PATH=/data/pulkitag/misc/raghuramkowdeed/projects/Self-Distillation
cd /data/pulkitag/misc/raghuramkowdeed/stream-bench
# REINFORCE++ on the FULL fused stream (4219) — 4th baseline line.
# 4h wall for backfill; resumes from checkpoint on preemption/timeout.
python3 -u run_fused_reinforce.py \
  --output_dir results/fused_reinforce_full \
  --stream_order fused_stream_order_full.pkl \
  --checkpoint_every 5
echo "DONE"
