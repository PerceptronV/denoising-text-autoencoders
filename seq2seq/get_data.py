import os
import argparse

import sys
sys.path.append('../')

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="../data/anki_parallel/eng_train.txt",
                    help='Filepath to data file')
parser.add_argument('--output', type=str, default="./seq_data/eng_train_d@_s#.txt",
                    help='Filepath to output file (@ for fraction, # for number of sentences)')
parser.add_argument('--fraction', type=float, default=1,
                    help='Fraction of data to use')


if __name__ == "__main__":
    args = parser.parse_args()

    sents = load_sent(args.input)

    size = len(sents)
    keep = int(size * args.fraction)

    print(f"Loaded {size} sentences, keeping {keep} for output")

    out_path = args.output
    out_path = out_path.replace("@", str(args.fraction))
    out_path = out_path.replace("#", str(keep))

    dirname = os.path.dirname(out_path)
    os.makedirs(dirname, exist_ok=True)

    write_sent(sents[:keep], out_path)

    print(f"Select number of data successfully written at `{out_path}`")
