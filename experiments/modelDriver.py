import os

prefix = "conda activate ml-high & python mapping.py"

for epochs in (10, 20, 30):
    for loss_func in ('mse', 'cosine'):
        for activation in ('relu', 'sigmoid', 'tanh'):
            for units in (64, 128, 256):
                for layers in (1, 2, 3, 4):
                    cmd = f"{prefix} --epochs {epochs} --loss-func {loss_func} --activation {activation} --layers {layers} --units {units}"
                    print(cmd)
                    os.system(cmd)
                    print("\n\n\n")
