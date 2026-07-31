#!/bin/bash
#SBATCH --job-name=fused_gft0
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
# Fused full stream (4219) with generate_from_teacher=False (paper default):
# completions generated from the BARE student prompt; teacher (hint) only supplies
# the KL target distribution. Everything else identical to results/fused_sdft.
# 4h wall for fast backfill; resumes from checkpoint on preemption/timeout.
python3 -u run_fused.py --method sdft \
  --output_dir results/fused_sdft_gft0 \
  --stream_order fused_stream_order_full.pkl \
  --checkpoint_every 5 --gen_from_teacher 0
echo "DONE"
