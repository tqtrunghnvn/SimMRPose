#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2
IMG_DIR=$3
if ! ([[ -n "${CONFIG}" ]] || [[ -n "${CHECKPOINT}" ]] || [[ -n "${IMG_DIR}" ]]); then
    echo "Argument CONFIG or CHECKPOINT or IMG_DIR is missing"
    exit
fi

python pose_estimation/test.py --cfg ${CONFIG} --model-file ${CHECKPOINT} --dir ${IMG_DIR}
