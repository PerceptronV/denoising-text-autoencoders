bert-score \
  -r "../data/anki_parallel/spa_test.txt" \
  -c "../experiments/texts/model_simplification/test_eng2spa_n1_lmse_u128_arelu_e150_b64_d0.1.txt" \
  --lang es

bert-score \
  -r "../data/anki_parallel/spa_test.txt" \
  -c "../experiments/texts/writeup/test_eng2spa_n2_lmse_u256_arelu_e200_b64_d0.1.txt" \
  --lang es

bert-score \
  -r "../data/anki_parallel/spa_test.txt" \
  -c "../seq2seq/seq_outs/0.1/eng2spa_test_full.txt" \
  --lang es

bert-score \
  -r "../data/anki_parallel/spa_test.txt" \
  -c "../seq2seq/seq_outs/1.0/eng2spa_test_full.txt" \
  --lang es

# ../seq2seq/seq_outs/eng2spa_test_full.txt
