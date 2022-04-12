import torch.nn as nn


class MappingModel(nn.Module):
    def __init__(self, dims, nlayers=1, units=128, activation=nn.ReLU):
        super(MappingModel, self).__init__()
        print(nlayers)
        if nlayers == 1:
            self.linmap = nn.Linear(dims, dims)
        elif nlayers == 2:
            self.linmap = nn.Sequential(
                nn.Linear(dims, units), activation(), nn.Linear(units, dims)
            )
        else:
            stack = (
                [nn.Linear(dims, units), activation()]
                + [nn.Linear(units, units), activation()] * (nlayers - 2)
                + [nn.Linear(units, dims)]
            )
            self.linmap = nn.Sequential(*stack)

    def forward(self, x):
        x = self.linmap(x)
        return x
