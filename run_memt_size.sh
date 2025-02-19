#!/bin/bash

mkdir -p results
mkdir -p results/diff_memt_size

make hrocksdb && rm bin/test_puts && make bin/test_puts

memt_sizes=(10000 100000 500000 1000000 5000000 10000000)

for memt_size in "${memt_sizes[@]}"
do
    for key_size in 8 16
    do
        for val_size in 8 100
        do
            ./bin/test_puts -n 10000000 -k $key_size -v $val_size -m $memt_size > results/diff_memt_size/puts_10M_${memt_size}_${key_size}_${val_size}.txt
        done
    done
done
