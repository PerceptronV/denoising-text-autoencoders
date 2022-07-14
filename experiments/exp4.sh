logdir="runs/shrinking_dry_run"

loss_func='mse'
activation='relu'
units=128
layers=3

batch_size=64
fracs=(0.1 0.05 0.025 0.01)
eps=(60 58 60 77)

for i in ${!fracs[@]}
do
    data_fraction=${fracs[$i]}
    epochs=${eps[$i]}
    python mapping.py \
        --data-fraction $data_fraction \
        --epochs $epochs \
        --loss-func $loss_func \
        --activation $activation \
        --units $units \
        --layers $layers \
        --batch-size $batch_size \
        --log-dir $logdir
done
