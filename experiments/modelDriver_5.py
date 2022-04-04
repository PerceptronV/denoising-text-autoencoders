import os

prefix = "conda activate ml-high & python mapping.py"
logdir = "runs/model_simplification"

loss_func = 'mse'
activation = 'relu'
stopping = 0.01

batch_size = 64
k = 15 * 1

for data_fraction in (0.1, 0.05, 0.025, 0.01):
    for units in (128, 64, 32):
        for layers in (3, 2, 1):
            epochs = int(k / data_fraction)

            cmd = f"{prefix} --data-fraction {data_fraction} --units {units} --layers {layers} --loss-func {loss_func} --activation {activation} --epochs {epochs} --batch-size {batch_size} --early-stopping {stopping} --log-dir {logdir}"
            print(cmd)
            os.system(cmd)
            print("\n\n\n")
