from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from matplotlib import pyplot as plt
from statistics import stdev, mean, median, mode
from tqdm import tqdm
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--reference', type=str, required=True,
                    help='path to reference file')
parser.add_argument('-a', '--hypA', type=str, required=True,
                    help='path to hypothesis A file')
parser.add_argument('-b', '--hypB', type=str, required=True,
                    help='path to hypothesis B file')
parser.add_argument('-d', '--histdir', type=str, required=False, default="./histogram",
                    help='directory to store histogram')

def chunk2sents(chunk):
  chunk = chunk.strip()
  chunk = chunk.replace('\r', '')
  return chunk.split("\n")

def sents2wordlists(sents, is_ref):
  if is_ref:
    return [[[w for w in s.strip().split(" ")]] for s in sents]
  else:
    return [[w for w in s.strip().split(" ")] for s in sents]

def load_wordlists(fp, is_ref=False):
  with open(fp, 'r', encoding="utf-8") as f:
    chunk = f.read()
  return sents2wordlists(chunk2sents(chunk), is_ref)

def multi_sentence_bleu(refs, hyps):
  ret = []
  for (r, h) in tqdm(zip(refs, hyps)):
    ret.append(sentence_bleu(r, h))
  return ret

def count(l, func):
  ret = 0
  for i in l:
    if func(i):
      ret += 1
  return ret

def stats(m_bleus, c_val):
  ret  = f"micro-average mean | {c_val}\n"
  ret += f"macro-average mean | {mean(m_bleus)}\n"
  ret += f"            median | {median(m_bleus)}\n"
  ret += f"              mode | {mode(m_bleus)}\n"
  ret += f"standard deviation | {stdev(m_bleus)}\n"
  ret += f"           # == 0s | {count(m_bleus, lambda x : x == 0)}\n"
  ret += f"         # <= 0.1s | {count(m_bleus, lambda x : x < 0.1)}\n"
  ret += f"         # <= 0.4s | {count(m_bleus, lambda x : x <= 0.4)}\n"
  ret += f"          # > 0.4s | {count(m_bleus, lambda x : x > 0.4)}\n"
  ret += f"           # == 1s | {count(m_bleus, lambda x : x == 1)}\n"
  return ret

def histogram(vals, title, fname, bins=150):
  n, bin_edges, _ = plt.hist(vals, bins=bins)
  plt.title(title)
  plt.xlim([0, 1])
  plt.ylim([0, 200])
  plt.ylabel('Frequency')
  plt.xlabel('BLEU score')
  plt.show()
  plt.savefig(os.path.join(args.histdir, fname+'.png'))
  
  n_sort = n.copy()
  n_sort.sort()
  mode = n_sort[-3]
  mode_idx = np.where(n == mode)[-1]
  edge = bin_edges[mode_idx]

  print(mode, edge, bin_edges)

if __name__ == "__main__":

  args = parser.parse_args()

  ref = load_wordlists(args.reference, is_ref=True)
  hypA = load_wordlists(args.hypA)
  hypB = load_wordlists(args.hypB)

  print(len(ref), len(hypA), len(hypB))

  print("Computing BLEU scores for hypothesis A...")
  hypA_multi_bleus = multi_sentence_bleu(ref, hypA)
  hypA_corpus_bleu_val = corpus_bleu(ref, hypA)
  print("Computing BLEU scores for hypothesis B...")
  hypB_multi_bleus = multi_sentence_bleu(ref, hypB)
  hypB_corpus_bleu_val = corpus_bleu(ref, hypB)
  
  print("\n")

  print(f"Hypothesis A BLEU score (from file at {args.hypA}):")
  print(stats(hypA_multi_bleus, hypA_corpus_bleu_val))
  print(f"Hypothesis B BLEU score: (from file at {args.hypB}):")
  print(stats(hypB_multi_bleus, hypB_corpus_bleu_val))

  histogram(hypA_multi_bleus, f"HypA BLEU Distibution ({args.hypA})", "hypA")
  histogram(hypB_multi_bleus, f"HypB BLEU Distibution ({args.hypB})", "hypB")
