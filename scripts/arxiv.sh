#!bin/bash

for (( i=1 ; i<=4 ; i++ ))
do
  touch logs/arxiv_log_$i
  echo "Start run $i"
  python main.py --cfg configs/GRIT/arxiv-GRIT-RRWP.yaml  wandb.use False accelerator "cpu" optim.max_epoch 50 seed $i dataset.dir ./datasets > logs/arxiv_log_$i
  echo "End run $i"
done