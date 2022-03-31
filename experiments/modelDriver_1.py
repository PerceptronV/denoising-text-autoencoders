import os

prefix = "conda activate ml-high & python mapping.py"
logdir = "runs/big_idea"

epochs = 15
batch_size = 64
data_fraction = 1

for loss_func in ('mse', 'cosine'):
    for activation in ('relu', 'sigmoid', 'tanh'):
        for units in (64, 128, 256):
            for layers in (1, 2, 3, 4):
                cmd = f"{prefix} --loss-func {loss_func} --activation {activation} --units {units} --layers {layers} --epochs {epochs} --batch-size {batch_size} --data-fraction {data_fraction} --log-dir {logdir}"
                print(cmd)
                os.system(cmd)
                print("\n\n\n")
