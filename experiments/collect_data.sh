# start in main directory
cd ../

python test.py --reconstruct --data data/anki_parallel/eng_train.txt --output_dir experiments/eng_reconstruct --output train --checkpoint checkpoints --verbose
python test.py --reconstruct --data data/anki_parallel/spa_train.txt --output_dir experiments/spa_reconstruct --output train --checkpoint checkpoints
