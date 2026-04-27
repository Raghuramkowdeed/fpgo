#!/bin/bash
#SBATCH --job-name=fgpo_import
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=lab-free
#SBATCH --account=csail
#SBATCH --time=00:05:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.out
#SBATCH --error=/data/pulkitag/misc/raghuramkowdeed/slurm_logs/%j.err

source /data/pulkitag/misc/raghuramkowdeed/setup_conda.sh myenv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
source ~/.bashrc

cd /data/pulkitag/misc/raghuramkowdeed/projects/agl

python -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import vllm.sampling_params as _vsp
if not hasattr(_vsp, 'GuidedDecodingParams'):
    _vsp.GuidedDecodingParams = None
from trl import RLOOTrainer, RLOOConfig
from cumreg.datasets.base import Problem
from cumreg.oracles.code_oracle import CodeOracle
from fgpo.frontier_client import FrontierClient
print('ALL IMPORTS OK')
print('ANTHROPIC_API_KEY set:', bool(os.environ.get('ANTHROPIC_API_KEY')))
"
