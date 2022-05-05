import torch

def tokenise(x, vocab):
    batch = []
    for s in x:
        tokens = [vocab.go] + [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s] + [vocab.eos]
        t = torch.tensor(tokens, dtype=torch.long)
        batch.append(t)
    return batch

def de_tokenise(y, vocab):
    omit_tokens = (vocab.go, vocab.pad)
    word_sents = []

    for s in y:
        word_s = []
        for i in s:
            if i == vocab.eos:
                break
            elif i not in omit_tokens:
                word_s.append(vocab.idx2word[i])
        word_sents.append(word_s)
    
    return word_sents
