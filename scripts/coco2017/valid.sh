#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2
COCO_BBOX_FILE=$3
if ! ([[ -n "${CONFIG}" ]] || [[ -n "${CHECKPOINT}" ]]); then
    echo "Argument CONFIG or CHECKPOINT is missing"
    exit
fi

if [[ -n "${COCO_BBOX_FILE}" ]]; then
    # Using the predicted human boxes from COCO_BBOX_FILE
    python pose_estimation/valid.py --cfg ${CONFIG} --flip-test --model-file ${CHECKPOINT} --use-detect-bbox --coco-bbox-file ${COCO_BBOX_FILE}
else
    # Using the ground-truth human boxes
    python pose_estimation/valid.py --cfg ${CONFIG} --flip-test --model-file ${CHECKPOINT}
fi
