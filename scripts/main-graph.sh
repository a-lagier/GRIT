#!/bin/bash
#SBATCH --account=def-coama74
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --mem-per-cpu=32G
#SBATCH --output=graph.out
#SBATCH --cpus-per-task=1

module load python/3.10 cuda scipy-stack
source ./env/bin/activate

bash scripts/molbace-GPS-ROGPE.sh

echo "Finish job"