import os

prefix = "conda activate ml-high & python mapping.py"
logdir = "./runs/shrinking_dry_run"

loss_func = 'mse'
activation = 'sigmoid'
units = 128
layers = 3

batch_size = 64
fracs = (0.1, 0.05, 0.025, 0.01)
eps = (60, 58, 60, 77)

for data_fraction, epochs in zip(fracs, eps):
    cmd = f"{prefix} --data-fraction {data_fraction} --loss-func {loss_func} --activation {activation} --units {units} --layers {layers} --epochs {epochs} --batch-size {batch_size} --log-dir {logdir}"
    print(cmd)
    os.system(cmd)
    print("\n\n\n")
