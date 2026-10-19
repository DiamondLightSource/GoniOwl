#!/bin/bash
#SBATCH --job-name=i23pinml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4
#SBATCH --partition=cs05r
#SBATCH --time=23:59:59
#SBATCH --output=./fromslurm.log

# Submit from the repository root after `uv venv && uv pip install -e .`:
#   sbatch src/Training/train_slurm.sh /path/to/goniopin_auto_<date>_binary
# Interactive alternative on Diamond's cluster:
#   ssh hopper
#   srun --nodes=1 --partition=cs05r --ntasks-per-node=20 --gres=gpu:4 --pty bash

source .venv/bin/activate
python src/Training/train_binary_classification_model.py --train-dir "$1" --log-file ./outputs/run.log
