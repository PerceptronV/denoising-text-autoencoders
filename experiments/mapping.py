import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mapModel import MappingModel


parser = argparse.ArgumentParser()
parser.add_argument('--vector-dir', type=str, default="./vectors", 
                    help='Directory to the train, test, and validation vector files for both English and Spanish')
parser.add_argument('--output-dir', type=str, default="./models", 
                    help='Directory to store the trained model')
parser.add_argument('--log-dir', type=str, default="./runs", 
                    help='Directory to store the tensorboard logs')
parser.add_argument('--seed', type=int, default=1920,
                    help="Seed for reproducibility")

parser.add_argument('--vector-dim', type=int, default=128,
                    help='Dimension of vector')
parser.add_argument('--epochs', type=int, default=15,
                    help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=64, 
                    help='Number of batches')
parser.add_argument('--data-fraction', type=float, default=1,
                    help='Fraction of data to use')
parser.add_argument('--loss-func', type=str, default='mse', choices=['mse', 'cosine'],
                    help='Type of loss function to use')
parser.add_argument('--layers', type=int, default=3,
                    help='Number of layers of mapping model')
parser.add_argument('--units', type=int, default=128,
                    help='Number of units in mapping model')
parser.add_argument('--activation', type=str, default='sigmoid', choices=['relu', 'sigmoid', 'tanh'],
                    help='Type of loss function to use')


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

        writer.add_scalar('Loss/Train', loss, writerStep)
        writerStep += 1

        if (step % (size // 5)) < BATCH_SIZE:
            print(f"loss: {loss:>7f}  [{step:>5d}/{size:>5d}]")
    
    mean_loss = np.mean(losses)
    print(f"> Mean loss: {mean_loss:>7f}")

    return mean_loss, writerStep

def valid_loop(eng_dataloader, spa_dataloader, model, loss_fn, writerStep):
    losses = []

    with torch.no_grad():
        for eng_vec, spa_vec in zip(eng_dataloader, spa_dataloader):
            pred = model(eng_vec)

            if LOSS_FUNC == "mse":
                loss = loss_fn(pred, spa_vec)
            elif LOSS_FUNC == "cosine":
                loss = -loss_fn(pred, spa_vec).abs().mean() # use this for cosine loss
            
            loss = loss.item()
            losses.append(loss)
    
    mean_loss = np.mean(losses)
    writer.add_scalar('Loss/Validation', mean_loss, writerStep)
    print(f"> Mean validation loss: {mean_loss:>7f}")

    return mean_loss

def load_parallel_data(dir, type, device, frac):
    eng_path = os.path.join(dir, f"eng_{type}.z.pt")
    spa_path = os.path.join(dir, f"spa_{type}.z.pt")

    eng_vectors = torch.tensor(torch.load(eng_path), dtype=torch.float32).to(device)
    spa_vectors = torch.tensor(torch.load(spa_path), dtype=torch.float32).to(device)

    size = len(eng_vectors)
    keep = int(size * frac)
    print(f"Loaded {size} {type} sentences; keeping {keep} {type} sentences")

    eng_vectors, spa_vectors = eng_vectors[:keep], spa_vectors[:keep]

    eng_dataloader = DataLoader(eng_vectors, batch_size=BATCH_SIZE, shuffle=False)
    spa_dataloader = DataLoader(spa_vectors, batch_size=BATCH_SIZE, shuffle=False)

    return eng_dataloader, spa_dataloader, keep


if __name__ == "__main__":
    args = parser.parse_args()

    VECTOR_DIM = args.vector_dim
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LOSS_FUNC = args.loss_func
    NLAYERS = args.layers
    UNITS = args.units
    ACTIVATION = args.activation

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (eng_train_dataloader, 
     spa_train_dataloader, 
                    keep)   = load_parallel_data(args.vector_dir, "train", device, args.data_fraction)
    (eng_valid_dataloader, 
     spa_valid_dataloader, 
                       _)  = load_parallel_data(args.vector_dir, "valid", device, 1)
    
    signature = f"n{NLAYERS}_l{LOSS_FUNC}_u{UNITS}_a{ACTIVATION}_e{EPOCHS}_b{BATCH_SIZE}_d{args.data_fraction}"

    afunc = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}[ACTIVATION]
    model = MappingModel(VECTOR_DIM, NLAYERS, activation=afunc).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if LOSS_FUNC == "mse":
        loss_fn = nn.MSELoss()
    elif LOSS_FUNC == "cosine":
        loss_fn = nn.CosineSimilarity() # use this for cosine loss
    
    writer = SummaryWriter(os.path.join(args.log_dir, f"log_{signature}"))
    writerStep = 0

    for t in range(EPOCHS):
        print(f"\nEpoch {t+1}\n-------------------------------")
        _, writerStep = train_loop(eng_train_dataloader, spa_train_dataloader, model, loss_fn, optimizer, writerStep)
        _ = valid_loop(eng_valid_dataloader, spa_valid_dataloader, model, loss_fn, writerStep)
    
    print("\nTraining complete")

    writer.close()
    torch.save(model, os.path.join(args.output_dir, f"model_{signature}.pt"))
    print("Model saved")
