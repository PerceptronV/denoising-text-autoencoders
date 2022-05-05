python seq_test.py --input-file ./seq_data/eng_test_d0.2_s5537.txt \
                   --output-file ./seq_outs/0.1/eng2spa_test_d0.2_s5537.txt \
                   --states-file ./ckpts/0.1/states.pt \
                   --vocab-file ../checkpoints/vocab.txt

python seq_test.py --input-file ../data/anki_parallel/eng_test.txt \
                   --output-file ./seq_outs/0.1/eng2spa_test_full.txt \
                   --states-file ./ckpts/0.1/states.pt \
                   --vocab-file ../checkpoints/vocab.txt

python seq_test.py --input-file ../data/anki_parallel/eng_test.txt \
                   --output-file ./seq_outs/1.0/eng2spa_test_full.txt \
                   --states-file ./ckpts/1.0/states.pt \
                   --vocab-file ../checkpoints/vocab.txt
