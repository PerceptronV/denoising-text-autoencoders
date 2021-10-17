import argparse
import numpy as np
import unicodedata
import os
import re

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--input', metavar='FILE', required=True,
                    help='path to input file')
parser.add_argument('--output', metavar='PATH', required=True,
                    help='path to output directory')
# Preprocessing arguments
parser.add_argument('--train', type=float, default=0.7, 
                    help='proportion of data used for training')
parser.add_argument('--test', type=float, default=0.2, 
                    help='proportion of data used for testing')
parser.add_argument('--valid', type=float, default=0.1, 
                    help='proportion of data used for validating')
parser.add_argument('--arith_eng_train', type=int, default=100, 
                    help='number of english sentences retained for training latent space arithmetic')
parser.add_argument('--arith_spa_train', type=int, default=100, 
                    help='number of spanish sentences retained for training latent space arithmetic')
parser.add_argument('--arith_eng_test', type=int, default=100, 
                    help='number of english sentences retained for testing latent space arithmetic')
parser.add_argument('--arith_spa_test', type=int, default=100, 
                    help='number of spanish sentences retained for testing latent space arithmetic')
# Others
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')

def preproc_sent(sent):
    # remove the funny stuff above some spanish characters
    # from a deprecated tensorflow tutorial
    ret = ''.join(c for c in unicodedata.normalize('NFD', sent)
                  if unicodedata.category(c) != 'Mn')
    
    ret = ret.lower().strip()

    # Turn numbers into `$_num_`
    ret = re.sub(r'[0-9]+', "$_num_", ret)

    # creating spaces between words and punctuations
    # from https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    ret = re.sub(r"([¿?.,;'\"$&!¡()])", r" \1 ", ret)
    ret = re.sub(r'[" "]+', " ", ret)

    # replace everything else with spaces
    ret = re.sub(r'[^a-zA-Z¿?.,;\'"$&!¡()_]+', " ", ret)

    return ret.strip()

def writefile(fpath, sents):
    dirs, _ = os.path.split(fpath)
    os.makedirs(dirs, exist_ok=True)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sents))

def main(args):
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    lines = text.split('\n')
    print("Number of lines: {}".format(len(lines)))
    
    eng_sents = []
    spa_sents = []
    for l in lines:
        eng_s, spa_s, _ = l.split('\t')
        eng_sents.append(preproc_sent(eng_s))
        spa_sents.append(preproc_sent(spa_s))
    
    eng_sents, spa_sents = np.array(eng_sents), np.array(spa_sents)
    np.random.shuffle(eng_sents)
    np.random.shuffle(spa_sents)

    total_sents_num = len(eng_sents) + len(spa_sents)

    (arith_eng_train_sents, 
     arith_eng_test_sents, 
     eng_remaining_sents) = (eng_sents[:args.arith_eng_train], 
                             eng_sents[args.arith_eng_train:args.arith_eng_train+args.arith_eng_test], 
                             eng_sents[args.arith_eng_train+args.arith_eng_test:])
    (arith_spa_train_sents, 
     arith_spa_test_sents, 
     spa_remaining_sents) = (spa_sents[:args.arith_spa_train], 
                             spa_sents[args.arith_spa_train:args.arith_spa_train+args.arith_spa_test], 
                             spa_sents[args.arith_spa_train+args.arith_spa_test:])
    
    remaining_sents = np.array(eng_remaining_sents.tolist() + spa_remaining_sents.tolist())
    np.random.shuffle(remaining_sents)

    num_total = len(remaining_sents)
    num_train = int(num_total * args.train)
    num_test = int(num_total * args.test)
    num_valid = int(num_total * args.valid)

    (train_sents, 
     test_sents, 
     valid_sents) = (remaining_sents[:num_train],
                     remaining_sents[num_train:num_train+num_test],
                     remaining_sents[num_train+num_test:num_train+num_test+num_valid])
    
    writefile(
        os.path.join(args.output, "language", "eng_{}_train.txt".format(args.arith_eng_train)), 
        arith_eng_train_sents.tolist()
    )
    writefile(
        os.path.join(args.output, "language", "spa_{}_train.txt".format(args.arith_spa_train)), 
        arith_spa_train_sents.tolist()
    )
    writefile(
        os.path.join(args.output, "language", "eng_{}_test.txt".format(args.arith_eng_test)), 
        arith_eng_test_sents.tolist()
    )
    writefile(
        os.path.join(args.output, "language", "spa_{}_test.txt".format(args.arith_spa_test)), 
        arith_spa_test_sents.tolist()
    )

    writefile(os.path.join(args.output, "train.txt"), train_sents.tolist())
    writefile(os.path.join(args.output, "test.txt"), test_sents.tolist())
    writefile(os.path.join(args.output, "valid.txt"), valid_sents.tolist())

    desc = """{} sentences processed in total.

{} English sentences used for arithmetic training, {} English sentences used for arithmetic testing.
{} Spanish sentences used for arithmetic training, {} Spanish sentences used for arithmetic testing.

{} sentences then remain.
Of these, {} sentences used for training, {} sentences used for testing, {} sentences used for validating.""".format(
    total_sents_num, args.arith_eng_train, args.arith_eng_test, args.arith_spa_train, args.arith_spa_test, 
    num_total, num_train, num_test, num_valid
    )

    print(desc)


if __name__ == '__main__':   
    args = parser.parse_args()
    np.random.seed = args.seed
    main(args)
