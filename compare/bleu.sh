python bleu.py \
  -r "../data/anki_parallel/spa_test.txt" \
  -a "../experiments/texts/model_simplification/test_eng2spa_n1_lmse_u128_arelu_e150_b64_d0.1.txt" \
  -b "../seq2seq/seq_outs/0.1/eng2spa_test_full.txt"

python bleu.py \
  -r "../data/anki_parallel/spa_test.txt" \
  -a "../experiments/texts/model_simplification/test_eng2spa_n1_lmse_u128_arelu_e150_b64_d0.1.txt" \
  -b "../seq2seq/seq_outs/1.0/eng2spa_test_full.txt"

# ../seq2seq/seq_outs/eng2spa_test_full.txt
