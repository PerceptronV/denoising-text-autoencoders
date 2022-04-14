import torch

def tokenise(x, vocab):
    batch = []
    for s in x:
        tokens = [vocab.go] + [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s] + [vocab.eos]
        t = torch.tensor(tokens, dtype=torch.long)
        batch.append(t)
    return batch

def de_tokenise(y, vocab):
    omit_tokens = (vocab.go, vocab.eos, vocab.pad)
    return [[vocab.idx2word[i] for i in s if i not in omit_tokens] for s in y]
