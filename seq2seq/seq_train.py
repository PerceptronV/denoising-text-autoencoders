import os
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from seq_model import *
from logs import Logger

import sys
sys.path.append('../')
from utils import *
from vocab import Vocab


# :-------------------------------------------------------------------------------------: #
# :-------------------------------------------------------------------------------------: #
# :-------------------------------------------------------------------------------------: #

parser = argparse.ArgumentParser()
# File paths
parser.add_argument('--src-train', type=str,
                    help='Filepath to source train file')
parser.add_argument('--trg-train', type=str,
                    help='Filepath to target train file')
parser.add_argument('--src-valid', type=str,
                    help='Filepath to source valid file')
parser.add_argument('--trg-valid', type=str,
                    help='Filepath to target valid file')
parser.add_argument('--ckpt-dir', type=str, default="./ckpts",
                    help='Directory to store the checkpoints and logs')
parser.add_argument('--vocab-file', type=str,
                    help='Filepath to vocab file')
parser.add_argument('--load-existing', action="store_true",
                    help='Whether or not to load any existing checkpoint')

# Architecture arguments
parser.add_argument('--enc-emb-dim', type=int, default=128,
                    help='Encoder embedding dimensions')
parser.add_argument('--dec-emb-dim', type=int, default=128,
                    help='Decoder embedding dimensions')
parser.add_argument('--enc-hid-dim', type=int, default=256,
                    help='Encoder hidden dimensions')
parser.add_argument('--dec-hid-dim', type=int, default=256,
                    help='Deocder hidden dimensions')
parser.add_argument('--attn-dim', type=int, default=32,
                    help='Attention dimenions')

# Training arguments
parser.add_argument('--enc-dropout', type=float, default=0.5,
                    help='Encoder dropout rate')
parser.add_argument('--dec-dropout', type=float, default=0.5,
                    help='Decoder dropout rate')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=256,
                    help='Batch size')
parser.add_argument('--clip', type=float, default=1,
                    help='Gradient clipping')

# Miscellaneous
parser.add_argument('--seed', type=int, default=1920,
                    help="Random seed for reproducibility")
parser.add_argument('--no-cuda', action="store_true",
                    help='Do not use CUDA')


# :-------------------------------------------------------------------------------------: #
# :-------------------------------------------------------------------------------------: #
# :-------------------------------------------------------------------------------------: #

def tokenise(x, vocab):
    batch = []
    for s in x:
        tokens = [vocab.go] + [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s] + [vocab.eos]
        t = torch.tensor(tokens, dtype=torch.long)
        batch.append(t)
    return batch


def generate_batch(data_batch):
    # Code modified from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    # © Copyright 2017, PyTorch.
    
    return pad_sequence(data_batch, padding_value=PAD_IDX).to(device)


def train(model: nn.Module,
          src_iterator: torch.LongTensor,
          trg_iterator: torch.LongTensor,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float,
          logging: Logger):
    # Code modified from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    # © Copyright 2017, PyTorch.

    model.train()
    epoch_loss = 0

    for _, (src, trg) in enumerate(zip(src_iterator, trg_iterator)):
        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        logging.write_scalar("Loss/Train", loss.item())

    return epoch_loss / len(src_iterator)


def evaluate(model: nn.Module,
             src_iterator: torch.LongTensor,
             trg_iterator: torch.LongTensor,
             criterion: nn.Module,
             logging: Logger):
    # Code modified from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    # © Copyright 2017, PyTorch.

    model.eval()
    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(zip(src_iterator, trg_iterator)):
            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(src_iterator)
    logging.write_scalar("Loss/Valid", avg_loss, increment=False)
    return avg_loss


def epoch_time(start_time: int,
               end_time: int):
    # Code from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    # © Copyright 2017, PyTorch.

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# :-------------------------------------------------------------------------------------: #
# :-------------------------------------------------------------------------------------: #
# :-------------------------------------------------------------------------------------: #

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)
    
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    set_seed(args.seed)

    logging = Logger(os.path.join(args.ckpt_dir, "logs"))

    # Prepare data
    vocab = Vocab(args.vocab_file)
    PAD_IDX = vocab.pad

    src_train_sents = tokenise(load_sent(args.src_train), vocab)
    trg_train_sents = tokenise(load_sent(args.trg_train), vocab)
    src_valid_sents = tokenise(load_sent(args.src_valid), vocab)
    trg_valid_sents = tokenise(load_sent(args.trg_valid), vocab)
    
    # Following code modified from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    # © Copyright 2017, PyTorch.

    src_train_batches = DataLoader(src_train_sents, batch_size=args.batch_size,
                                   shuffle=False, collate_fn=generate_batch)
    trg_train_batches = DataLoader(trg_train_sents, batch_size=args.batch_size,
                                   shuffle=False, collate_fn=generate_batch)
    src_valid_batches = DataLoader(src_valid_sents, batch_size=args.batch_size,
                                   shuffle=False, collate_fn=generate_batch)
    trg_valid_batches = DataLoader(trg_valid_sents, batch_size=args.batch_size,
                                   shuffle=False, collate_fn=generate_batch)

    # Define model
    INPUT_DIM = OUTPUT_DIM = vocab.size
    ENC_EMB_DIM = args.enc_emb_dim
    DEC_EMB_DIM = args.dec_emb_dim
    ENC_HID_DIM = args.enc_hid_dim
    DEC_HID_DIM = args.dec_hid_dim
    ATTN_DIM = args.attn_dim
    ENC_DROPOUT = args.enc_dropout
    DEC_DROPOUT = args.dec_dropout

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    # Optim & Loss
    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad)

    # Training
    N_EPOCHS = args.epochs
    CLIP = args.clip

    states_ckpt_path = os.path.join(args.ckpt_dir, 'states.pt')
    progress_ckpt_path = os.path.join(args.ckpt_dir, 'progress.pt')

    if args.load_existing and os.path.exists(states_ckpt_path) and os.path.exists(progress_ckpt_path):
        states = torch.load(states_ckpt_path)
        best_valid_loss = states['best_valid']
        model.load_state_dict(states['model_state_dict'])
        
        progress = torch.load(progress_ckpt_path)
        ep_range = range(progress['epoch'] + 1, N_EPOCHS)
        logging.set_global_step(progress['global_step'])
        optimizer.load_state_dict(progress['optimizer_state_dict'])
        
    else:
        ep_range = range(N_EPOCHS)
        best_valid_loss = float('inf')
        logging(f'The model has {count_parameters(model):,} trainable parameters')


    for epoch in ep_range:

        start_time = time.time()

        train_loss = train(model, src_train_batches, trg_train_batches, optimizer, criterion, CLIP, logging)
        valid_loss = evaluate(model, src_valid_batches, trg_valid_batches, criterion, logging)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        logging(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logging(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        logging(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'best_valid': best_valid_loss,
                'model_state_dict': model.state_dict(),
            }, states_ckpt_path)
            logging('Best valid loss. Model saved.')
        
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'global_step': logging.gstep,
            'optimizer_state_dict': optimizer.state_dict()
        }, progress_ckpt_path)
    
    logging.flush()
