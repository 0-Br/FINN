#!/bin/bash

python="/home/binrui/miniconda3/envs/earthkit/bin/python"
train="./train.py"

${python} ${train} --config="./configs/test.yaml" --model="PINN" --version="test"
