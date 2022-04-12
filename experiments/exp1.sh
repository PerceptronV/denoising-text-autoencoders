logdir="runs/big_idea"

epochs=15
batch_size=64
data_fraction=1

for loss_func in "mse" "cosine"
do
    for activation in "relu" "sigmoid" "tanh"
    do
        for units in 64 128 256
        do
            for layers in 1 2 3 4
            do
                python mapping.py \
                --loss-func $loss_func \
                --activation $activation \
                --units $units \
                --layers $layers \
                --epochs $epochs \
                --batch-size $batch_size \
                --data-fraction $data_fraction \
                --log-dir $logdir
            done
        done
    done
done
