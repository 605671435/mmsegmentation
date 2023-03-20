CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py \
    $CONFIG &
CUDA_VISIBLE_DEVICES=1 python $(dirname "$0")/train.py \
    $CONFIG
