python get_data_fracs.py --input ../data/anki_parallel/eng_train.txt --output ./seq2seq_data/eng_train_d@_s#.txt --fraction 0.1
python get_data_fracs.py --input ../data/anki_parallel/spa_train.txt --output ./seq2seq_data/spa_train_d@_s#.txt --fraction 0.1

python get_data_fracs.py --input ../data/anki_parallel/eng_test.txt --output ./seq2seq_data/eng_test_d@_s#.txt --fraction 0.2
python get_data_fracs.py --input ../data/anki_parallel/spa_test.txt --output ./seq2seq_data/spa_test_d@_s#.txt --fraction 0.2

python get_data_fracs.py --input ../data/anki_parallel/eng_valid.txt --output ./seq2seq_data/eng_valid_d@_s#.txt --fraction 0.4
python get_data_fracs.py --input ../data/anki_parallel/spa_valid.txt --output ./seq2seq_data/spa_valid_d@_s#.txt --fraction 0.4
