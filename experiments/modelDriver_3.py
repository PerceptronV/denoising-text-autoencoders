import os

prefix = "conda activate ml-high & python mapping.py"
logdir = "runs/shrinking"

loss_func = 'mse'
activation = 'sigmoid'
units = 128
layers = 3

batch_size = 64

k = 15 * 1

for data_fraction in (1, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01):    
    epochs = int(k/ data_fraction)

    cmd = f"{prefix} --data-fraction {data_fraction} --loss-func {loss_func} --activation {activation} --units {units} --layers {layers} --epochs {epochs} --batch-size {batch_size} --log-dir {logdir}"
    print(cmd)
    os.system(cmd)
    print("\n\n\n")
