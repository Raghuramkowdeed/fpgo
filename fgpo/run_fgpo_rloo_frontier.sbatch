#!/bin/bash
#SBATCH --job-name=fgpo_frontier
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=lab-free
#SBATCH --account=csail
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.out
#SBATCH --error=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.err

source /data/pulkitag/misc/raghuramkowdeed/setup_conda.sh myenv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
source ~/.bashrc

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set" >&2
    exit 1
fi

cd /data/pulkitag/misc/raghuramkowdeed/projects/agl

OUT=/data/pulkitag/misc/raghuramkowdeed/projects/agl/fgpo/results_fgpo_rloo_frontier
RESUME_FLAG=""
if [ -f "$OUT/state.json" ]; then
    RESUME_FLAG="--resume"
fi

python -u fgpo/run_fgpo_rloo.py \
    --output_dir "$OUT" \
    --hint_cache_path /data/pulkitag/misc/raghuramkowdeed/projects/agl/fgpo/cache/hints_frontier.pkl \
    --use_frontier_hints \
    --frontier_fraction 0.5 \
    --batch_size 50 \
    --num_generations 10 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 10 \
    --generation_batch_size 100 \
    --max_new_tokens 2048 \
    --max_seq_length 4096 \
    --learning_rate 5e-5 \
    --beta 0.01 \
    --n_epochs 1 \
    --fgpo_n_rounds 3 \
    --fgpo_probe_samples 10 \
    --fgpo_stop_reward 1.0 \
    --fgpo_worst_k 2 \
    --fgpo_best_k 2 \
    --oracle_timeout 60 \
    --oracle_workers 16 \
    --test_eval_every 1 \
    --test_eval_n_problems 100 \
    --frontier_model claude-sonnet-4-6 \
    --seed 42 \
    $RESUME_FLAG
