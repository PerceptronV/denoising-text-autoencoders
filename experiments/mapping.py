import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mapModel import MappingModel

def train_loop(eng_dataloader, spa_dataloader, model, loss_fn, optimizer, writerStep):
    size = len(eng_dataloader.dataset)
    losses = []

    for batch, (eng_vec, spa_vec) in enumerate(zip(eng_dataloader, spa_dataloader)):
        step = batch * BATCH_SIZE

        pred = model(eng_vec)

        if LOSS_FUNC == "mse":
            loss = loss_fn(pred, spa_vec)
        elif LOSS_FUNC == "cosine":
            loss = -loss_fn(pred, spa_vec).abs().mean() # use this for cosine loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        losses.append(loss)

        writer.add_scalar('Train/Loss', loss, writerStep)
        writerStep += 1

        if (step % (size // 5)) < BATCH_SIZE:
            print(f"loss: {loss:>7f}  [{step:>5d}/{size:>5d}]")
    
    mean_loss = np.mean(losses)
    print(f"> Mean loss: {mean_loss:>7f}")

    return mean_loss, writerStep


VECTOR_DIM = 128
EPOCHS = 10
BATCH_SIZE = 64
LOSS_FUNC = "mse"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter(f"runs/log_{LOSS_FUNC}")
writerStep = 0

eng_vectors = torch.tensor(torch.load('eng_train.z.pt'), dtype=torch.float32).to(device)
spa_vectors = torch.tensor(torch.load('spa_train.z.pt'), dtype=torch.float32).to(device)

eng_dataloader = DataLoader(eng_vectors, batch_size=BATCH_SIZE, shuffle=False)
spa_dataloader = DataLoader(spa_vectors, batch_size=BATCH_SIZE, shuffle=False)

model = MappingModel(VECTOR_DIM)
# print(f"Shape of eng vectors: {eng_vectors.shape}")
# print(model(eng_vectors[0]))

loss_cosine = nn.CosineSimilarity()
loss_mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if LOSS_FUNC == "mse":
    loss_fn = loss_mse
elif LOSS_FUNC == "cosine":
    loss_fn = loss_cosine # use this for cosine loss

for t in range(EPOCHS):
    print(f"\nEpoch {t+1}\n-------------------------------")
    _, writerStep = train_loop(eng_dataloader, spa_dataloader, model, loss_fn, optimizer, writerStep)
print("\nTraining complete")

writer.close()

torch.save(model.state_dict(), f"model_{LOSS_FUNC}.pt")
