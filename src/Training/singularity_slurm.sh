#!/bin/bash
#SBATCH --job-name=i23pinml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --gres=gpu:4
#SBATCH --partition=cs05r
#SBATCH --time=23:59:59
#SBATCH --output=./fromslurm.log
 
tfimage=/path/to/tensorflow_2.8.2-gpu-jupyter.sif
 
singularity exec --nv --home $PWD $tfimage python /path/to/img_classification.py