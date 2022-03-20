# start in main directory
cd ../

python test.py --reconstruct --data data/anki_parallel/eng_train.txt --output_dir experiments/eng_reconstruct --output train --checkpoint checkpoints --verbose
mv experiments/eng_reconstruct/train.z.pt experiments/vectors/eng_train.z.pt
python test.py --reconstruct --data data/anki_parallel/eng_test.txt --output_dir experiments/eng_reconstruct --output test --checkpoint checkpoints --verbose
mv experiments/eng_reconstruct/test.z.pt experiments/vectors/eng_test.z.pt
python test.py --reconstruct --data data/anki_parallel/eng_valid.txt --output_dir experiments/eng_reconstruct --output valid --checkpoint checkpoints --verbose
mv experiments/eng_reconstruct/valid.z.pt experiments/vectors/eng_valid.z.pt
rm -r experiments/eng_reconstruct

python test.py --reconstruct --data data/anki_parallel/spa_train.txt --output_dir experiments/spa_reconstruct --output train --checkpoint checkpoints --verbose
mv experiments/spa_reconstruct/train.z.pt experiments/vectors/spa_train.z.pt
python test.py --reconstruct --data data/anki_parallel/spa_test.txt --output_dir experiments/spa_reconstruct --output test --checkpoint checkpoints --verbose
mv experiments/spa_reconstruct/test.z.pt experiments/vectors/spa_test.z.pt
python test.py --reconstruct --data data/anki_parallel/spa_valid.txt --output_dir experiments/spa_reconstruct --output valid --checkpoint checkpoints --verbose
mv experiments/spa_reconstruct/valid.z.pt experiments/vectors/spa_valid.z.pt
rm -r experiments/spa_reconstruct
