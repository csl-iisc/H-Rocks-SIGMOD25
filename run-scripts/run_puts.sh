#!/bin/bash

mkdir -p results
mkdir -p results/puts2

make hrocksdb && rm bin/test_puts && make bin/test_puts

keys=(10000 100000 500000 1000000 5000000 10000000 25000000 50000000 100000000)

for key in "${keys[@]}"
do
    for key_size in 8 16
    do
        for val_size in 8 100
        do
            ./bin/test_puts -n $key -k $key_size -v $val_size > results/puts2/puts_${key}_${key_size}_${val_size}.txt
        done
    done
done
