#!/bin/bash
#SBATCH --job-name=raytracing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
# #SBATCH --exclusive
#SBATCH --time=00:00:30
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=v100
#SBATCH --output=ray_tracing_output.txt
#SBATCH --error=ray_tracing_error.txt
#SBATCH --account=mpcs51087

mpirun ./ray_tracing_multi 1000000000 1000 1024 512
