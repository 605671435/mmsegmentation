CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

for var in 0 1 2 3 4
#for var in 0 1 2 3 4 5 6 7 8 9
#for var in 0 1 2
do
  python $(dirname "$0")/train.py \
  $CONFIG
  echo $var
done