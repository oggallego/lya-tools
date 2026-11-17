#!/bin/bash

for i in 0 1 2 3 4 5
do
    echo "Running Cobaya for kmax cut $i"
    python3 run_cobaya_kmax.py $i
done
