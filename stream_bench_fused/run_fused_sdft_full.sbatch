#!/bin/bash
#SBATCH --job-name=fused_sdft_full
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --account=csail
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.out
#SBATCH --error=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.err

# FULL-data fused SDFT = CONTINUATION of the small run. Same output_dir -> resumes
# from the small run's batch-90 checkpoint (first 900 identical), then streams the
# rest. Launch ONLY after the small fused_sdft (results/fused_sdft) reaches DONE.
source /data/pulkitag/misc/raghuramkowdeed/setup_conda.sh myenv
cd /data/pulkitag/misc/raghuramkowdeed/stream-bench
python run_fused.py --method sdft \
  --output_dir results/fused_sdft \
  --stream_order fused_stream_order_full.pkl \
  --checkpoint_every 5
