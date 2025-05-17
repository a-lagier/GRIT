#!bin/bash

for (( i=0 ; i<=1 ; i++ ))
do
  touch logs/moltoxcast_log_$((4*i + 1))
  touch logs/moltoxcast_log_$((4*i + 2))
  touch logs/moltoxcast_log_$((4*i + 3))
  touch logs/moltoxcast_log_$((4*i + 4))
  echo "Start run $i"
  python main.py --cfg configs/GRIT/moltoxcast-GRIT-RRWP.yaml  wandb.use False accelerator "cuda:0" seed $((4*i + 1)) optim.max_epoch 50  &
  python main.py --cfg configs/GRIT/moltoxcast-GRIT-RRWP.yaml  wandb.use False accelerator "cuda:1" seed $((4*i + 2)) optim.max_epoch 50  &
  python main.py --cfg configs/GRIT/moltoxcast-GRIT-RRWP.yaml  wandb.use False accelerator "cuda:2" seed $((4*i + 3)) optim.max_epoch 50  &
  python main.py --cfg configs/GRIT/moltoxcast-GRIT-RRWP.yaml  wandb.use False accelerator "cuda:3" seed $((4*i + 4)) optim.max_epoch 50  &
  wait
  echo "End run $i"
done