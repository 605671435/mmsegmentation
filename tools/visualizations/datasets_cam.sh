#!/bin/bash
dataset=(X1 X2 AB)
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
for var in 0 1 2
do
    python ../get_embedding/test.py --dataset ${dataset[$var]}
done