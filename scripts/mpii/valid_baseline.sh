#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2
if ! ([[ -n "${CONFIG}" ]] || [[ -n "${CHECKPOINT}" ]]); then
    echo "Argument CONFIG or CHECKPOINT is missing"
    exit
fi

python pose_estimation/valid_baseline.py --cfg ${CONFIG} --flip-test --model-file ${CHECKPOINT}
