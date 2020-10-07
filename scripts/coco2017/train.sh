#!/usr/bin/env bash
CONFIG=$1
if ! ([[ -n "${CONFIG}" ]]); then
    echo "Argument CONFIG is missing"
    exit
fi

python pose_estimation/train.py --cfg ${CONFIG}
