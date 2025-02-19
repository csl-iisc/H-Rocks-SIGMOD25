#!/bin/bash

mkdir -p results
mkdir -p results/gets2

make hrocksdb && rm bin/test_put_get && make bin/test_put_get

prefill_keys=100000000
keys=(10000 100000 500000 1000000 5000000 10000000 25000000 50000000 100000000)

for key in "${keys[@]}"
do
    for key_size in 8
    do
        for val_size in 8
        do
            ./bin/test_put_get -p $prefill_keys -g $key -k $key_size -v $val_size > results/gets2/gets_${key}_${key_size}_${val_size}.txt
        done
    done
done
