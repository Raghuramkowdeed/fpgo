#!/bin/bash
#SBATCH --job-name=fused_sdft
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=lab-free
#SBATCH --account=csail
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.out
#SBATCH --error=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.err

# Fused distribution-shift stream (900 = DS-1000 + DDXPlus + HotpotQA), method=sdft.
# Auto-resumes from results/fused_sdft/state.json if present.
source /data/pulkitag/misc/raghuramkowdeed/setup_conda.sh myenv
cd /data/pulkitag/misc/raghuramkowdeed/stream-bench
python run_fused.py --method sdft \
  --output_dir results/fused_sdft \
  --checkpoint_every 5
