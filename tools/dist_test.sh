#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=$((60000 + RANDOM % 1000))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$ROOT_DIR":"$ROOT_DIR/external":$PYTHONPATH

echo "ROOT_DIR: $ROOT_DIR"
echo "PYTHONPATH: $PYTHONPATH"

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    "$SCRIPT_DIR/test.py" $CONFIG --launcher pytorch ${@:3}
