logdir="runs/model_simplification"

loss_func='mse'
activation='relu'
stopping=0.0075
batch_size=64
k=$((20 * 1))

python mapping.py \
    --data-fraction 1 \
    --units 256 \
    --layers 3 \
    --loss-func mse \
    --activation relu \
    --epochs 20 \
    --batch-size 64 \
    --log-dir $logdir \
    --early-stopping $stopping

for data_fraction in 0.25 0.1 0.05
do
    for units in 256 128 64 32
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
