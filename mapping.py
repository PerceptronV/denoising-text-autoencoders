import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_loop(eng_dataloader, spa_dataloader, model, loss_fn, optimizer):
    size = len(eng_dataloader.dataset)
    for batch, (eng_vec, spa_vec) in enumerate(zip(eng_dataloader, spa_dataloader)):
        pred = model(eng_vec)
        loss = loss_fn(pred, spa_vec)
        print(loss.shape)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % (size // 5) == 0:
            loss, current = loss.item(), batch * len(eng_vec)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


VECTOR_DIM = 128
EPOCHS = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

eng_vectors = torch.tensor(torch.load("eng_train.z.pt"), dtype=torch.float32).to(device)
spa_vectors = torch.tensor(torch.load("spa_train.z.pt"), dtype=torch.float32).to(device)

eng_dataloader = DataLoader(eng_vectors, batch_size=64, shuffle=False)
spa_dataloader = DataLoader(spa_vectors, batch_size=64, shuffle=False)

class MappingModel(nn.Module):
    def __init__(self, dims):
        super(MappingModel, self).__init__()
        self.linmap = nn.Linear(dims, dims)
    
    def forward(self, x):
        x = self.linmap(x)
        return x

model = MappingModel(VECTOR_DIM)
print(f"Shape of eng vectors: {eng_vectors.shape}")
print(model(eng_vectors[0]))

loss_fn = nn.CosineSimilarity()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(eng_dataloader, spa_dataloader, model, loss_fn, optimizer)
print("Training complete")
