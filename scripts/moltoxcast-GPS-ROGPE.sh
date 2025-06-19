#!/bin/bash
#SBATCH --account=def-coama74
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --mem-per-cpu=32G
#SBATCH --output=moltoxcast_rogpe.out
#SBATCH --cpus-per-task=1

module load python/3.10 cuda scipy-stack
source ./env/bin/activate

python main.py --repeat 5 --cfg configs/GPS/moltoxcast-GPS-ROGPE.yaml  wandb.use False accelerator "cuda:0" optim.max_epoch 400 seed 10 dataset.dir ./datasets

echo "Finish job"
