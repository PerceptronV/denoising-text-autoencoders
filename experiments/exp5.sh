logdir="runs/model_simplification"

loss_func='mse'
activation='relu'
stopping=0.0075
batch_size=64
k=$((15 * 1))

for data_fraction in 0.1 0.05 0.025 0.01
do
    for units in 128 64 32
    do
        for layers in 3 2 1
        do
            epochs=`echo "scale=0; $k/$data_fraction" | bc`
            python mapping.py \
                --data-fraction $data_fraction \
                --units $units \
                --layers $layers \
                --loss-func $loss_func \
                --activation $activation \
                --epochs $epochs \
                --batch-size $batch_size \
                --log-dir $logdir \
                --early-stopping $stopping
        done
    done
done
