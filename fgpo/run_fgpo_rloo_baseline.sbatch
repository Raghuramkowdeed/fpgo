#!/bin/bash
#SBATCH --job-name=fgpo_baseline
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=lab-free
#SBATCH --account=csail
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.out
#SBATCH --error=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.err

source /data/pulkitag/misc/raghuramkowdeed/setup_conda.sh myenv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
source ~/.bashrc

cd /data/pulkitag/misc/raghuramkowdeed/projects/agl

OUT=/data/pulkitag/misc/raghuramkowdeed/projects/agl/fgpo/results_fgpo_rloo_baseline
RESUME_FLAG=""
if [ -f "$OUT/state.json" ]; then
    RESUME_FLAG="--resume"
fi

python -u fgpo/run_fgpo_rloo.py \
    --output_dir "$OUT" \
    --hint_cache_path /data/pulkitag/misc/raghuramkowdeed/projects/agl/fgpo/cache/hints_baseline.pkl \
    --batch_size 50 \
    --num_generations 10 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 10 \
    --generation_batch_size 200 \
    --max_new_tokens 2048 \
    --max_seq_length 4096 \
    --learning_rate 5e-5 \
    --beta 0.01 \
    --n_epochs 1 \
    --oracle_timeout 60 \
    --oracle_workers 8 \
    --test_eval_every 1 \
    --test_eval_n_problems 100 \
    --seed 42 \
    $RESUME_FLAG
