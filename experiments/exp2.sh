logdir="runs/hypers"

loss_func='mse'
activation='relu'
units=128
layers=2
data_fraction=1

for epochs in 10 20 30
do
    for batch_size in 32 64 128
    do
        python mapping.py \
            --epochs $epochs \
            --batch-size $batch_size \
            --loss-func $loss_func \
            --activation $activation \
            --units $units \
            --layers $layers \
            --data-fraction $data_fraction \
            --log-dir $logdir
    done
done
