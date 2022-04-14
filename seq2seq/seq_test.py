import os
import math
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from seq_model import *
from seq_utils import *
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
parser.add_argument('--input-file', type=str,
                    help='Filepath to input translation file')
parser.add_argument('--output-file', type=str,
                    help='Filepath to output translation file')
parser.add_argument('--states-file', type=str, default="./ckpts",
                    help='Filepath to checkpoint `states.pt` file')
parser.add_argument('--vocab-file', type=str,
                    help='Filepath to vocab file')

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

# Miscellaneous
parser.add_argument('--batch-size', type=int, default=256,
                    help='Batch size')
parser.add_argument('--max-len', type=int, default=30,
                    help="Maximum length of output sentence")
parser.add_argument('--seed', type=int, default=1920,
                    help="Random seed for reproducibility")
parser.add_argument('--no-cuda', action="store_true",
                    help='Do not use CUDA')

def generate_batch(data_batch):
    # Code modified from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    # © Copyright 2017, PyTorch.
    
    return pad_sequence(data_batch, padding_value=PAD_IDX).to(device)


# :-------------------------------------------------------------------------------------: #
# :-------------------------------------------------------------------------------------: #
# :-------------------------------------------------------------------------------------: #

if __name__ == "__main__":
    args = parser.parse_args()
    dirname = os.path.dirname(args.output_file)
    os.makedirs(dirname, exist_ok=True)
    
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    set_seed(args.seed)

    # Prepare data
    vocab = Vocab(args.vocab_file)
    PAD_IDX = vocab.pad
    src_sents = tokenise(load_sent(args.input_file), vocab)
    
    # Following code modified from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    # © Copyright 2017, PyTorch.

    src_batches = DataLoader(src_sents, batch_size=args.batch_size,
                             shuffle=False, collate_fn=generate_batch)

    # Define model
    INPUT_DIM = OUTPUT_DIM = vocab.size
    ENC_EMB_DIM = args.enc_emb_dim
    DEC_EMB_DIM = args.dec_emb_dim
    ENC_HID_DIM = args.enc_hid_dim
    DEC_HID_DIM = args.dec_hid_dim
    ATTN_DIM = args.attn_dim

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, dropout=0)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, dropout=0, attention=attn)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    states = torch.load(args.states_file, map_location=device)
    model.load_state_dict(states['model_state_dict'])

    model.eval()
    trg_sents = []

    with torch.no_grad():
        for batch in tqdm(src_batches):
            output = model.translate(batch, vocab.go, vocab.pad, args.max_len)
            sent = de_tokenise(output, vocab)
            trg_sents.extend(sent)
    
    write_sent(trg_sents, args.output_file)
