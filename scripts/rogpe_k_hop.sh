
for k in $(seq 5 20);
do
    python main.py --cfg configs/GRIT/molbace-ROGPE.yaml wandb.use False accelerator "cpu" seed 30 posenc_ROGPE.k_hop $k dataset.dir ./datasets/
done