### Diamond Specific ###
# To run on a specific partition with 4 GPUs and 20 CPUs per node, you can use the following command:

ssh hopper

srun --nodes=1 --partition=cs05r --ntasks-per-node=20 --gres=gpu:4 --pty bash

### Generic ###

# Load the singularity image:

tfimage=/path/to/tensorflow_2.8.2-gpu-jupyter.sif

# Run the python script:

singularity exec --nv --home $PWD $tfimage python /path/to/img_classification_binary.py