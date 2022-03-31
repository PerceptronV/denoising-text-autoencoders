import os

prefix = "conda activate ml-high & python mapping.py"
logdir = "runs/shrinking"

loss_func = 'mse'
activation = 'sigmoid'
units = 128
layers = 3

epochs = 10
batch_size = 64

for data_fraction in (0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01):
    cmd = f"{prefix} --data-fraction {data_fraction} --loss-func {loss_func} --activation {activation} --units {units} --layers {layers} --epochs {epochs} --batch-size {batch_size}"
    print(cmd)
    os.system(cmd)
    print("\n\n\n")
