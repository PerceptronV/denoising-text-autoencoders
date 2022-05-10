from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tqdm import tqdm
from statistics import stdev, mean

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--reference', type=str, required=True,
                    help='path to reference file')
parser.add_argument('-a', '--hypA', type=str, required=True,
                    help='path to hypothesis A file')
parser.add_argument('-b', '--hypB', type=str, required=True,
                    help='path to hypothesis B file')

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

def stats(m_bleus, c_val):
  ret  = f"mico-average mean  {c_val}\n"
  ret += f"macro-average mean {mean(m_bleus)}\n"
  ret += f"standard deviation {stdev(m_bleus)}\n"
  return ret

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
