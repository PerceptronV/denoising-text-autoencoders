import torch.nn as nn

class MappingModel(nn.Module):
    def __init__(self, dims):
        super(MappingModel, self).__init__()
        self.linmap = nn.Linear(dims, dims)
    
    def forward(self, x):
        x = self.linmap(x)
        return x
