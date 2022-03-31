import os

prefix = "conda activate ml-high & python mapping.py"
logdir = "runs/hypers"

loss_func = 'mse'
activation = 'sigmoid'
units = 128
layers = 3

data_fraction = 1

for epochs in (10, 20, 30):
    for batch_size in (32, 64, 128):
        cmd = f"{prefix} --epochs {epochs} --batch-size {batch_size} --loss-func {loss_func} --activation {activation} --units {units} --layers {layers} --data-fraction {data_fraction}"
        print(cmd)
        os.system(cmd)
        print("\n\n\n")
