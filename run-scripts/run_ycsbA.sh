#!/bin/bash

mkdir -p results
mkdir -p results/YCSB_A2
mkdir -p results/YCSB_B2
mkdir -p results/YCSB_C2


make hrocksdb && make bin/test_ycsbA
make hrocksdb && make bin/test_ycsbB
make hrocksdb && make bin/test_ycsbC

prefill_keys=100000000
keys=(10000 100000 500000 1000000 5000000 10000000 25000000 50000000 100000000)

for key in "${keys[@]}"
do
    for key_size in 16
    do
        for val_size in 100
        do
            ./bin/test_ycsbA -p $prefill_keys -n $key -k $key_size -v $val_size > results/YCSB_A2/ycsbA_${key}_${key_size}_${val_size}.txt
            ./bin/test_ycsbB -p $prefill_keys -n $key -k $key_size -v $val_size > results/YCSB_B2/ycsbB_${key}_${key_size}_${val_size}.txt
            ./bin/test_ycsbC -p $prefill_keys -n $key -k $key_size -v $val_size > results/YCSB_C2/ycsbC_${key}_${key_size}_${val_size}.txt
        done
    done
done
