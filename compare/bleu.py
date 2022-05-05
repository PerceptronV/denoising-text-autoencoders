from nltk.translate.bleu_score import corpus_bleu
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

if __name__ == "__main__":

  args = parser.parse_args()

  ref = load_wordlists(args.reference, is_ref=True)
  hypA = load_wordlists(args.hypA)
  hypB = load_wordlists(args.hypB)

  print(len(ref), len(hypA), len(hypB))

  print("Computing BLEU scores for hypothesis A...")
  hypA_bleu = corpus_bleu(ref, hypA)
  print("Computing BLEU scores for hypothesis B...")
  hypB_bleu = corpus_bleu(ref, hypB)
  
  print("\n")

  print(f"Hypothesis A score: {hypA_bleu} (from file at {args.hypA})")
  print(f"Hypothesis B score: {hypB_bleu} (from file at {args.hypB})")

