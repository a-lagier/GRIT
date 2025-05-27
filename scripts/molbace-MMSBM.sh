#!bin/bash

for (( i=1 ; i<=5 ; i++ ))
do
  touch logs/molbace_log_$i
  echo "Start run $i"
  python main.py --cfg configs/GRIT/molbace-GRIT-MMSBM.yaml  wandb.use False accelerator "cpu" optim.max_epoch 100 seed $i dataset.dir ./datasets > logs/molbace_log_$i
  echo "End run $i"
done