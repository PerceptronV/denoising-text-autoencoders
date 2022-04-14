logdir="runs/shrinking"

loss_func='mse'
activation='relu'
units=128
layers=2
batch_size=64

k=$((15 * 1))
stopping=0.0075

for data_fraction in 1 0.9 0.75 0.5 0.25 0.1 0.05 0.025 0.01
do
    epochs=`echo "scale=0; $k/$data_fraction" | bc`
    python mapping.py \
        --data-fraction $data_fraction \
        --loss-func $loss_func \
        --activation $activation \
        --units $units \
        --layers $layers \
        --epochs $epochs \
        --batch-size $batch_size \
        --log-dir $logdir \
        --early-stopping $stopping
done
