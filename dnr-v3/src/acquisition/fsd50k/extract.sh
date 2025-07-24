#!/bin/bash

FSD50K_PATH="$RAW_DATA_ROOT/fsd50k"

LOG_PATH="$RAW_DATA_ROOT/fsd50k/_log"


nohup unzip -d "$FSD50K_PATH" "$FSD50K_PATH/FSD50K.ground_truth.zip" \
  > "$LOG_PATH/fsd-gt.out" 2>&1 &

nohup unzip -d "$FSD50K_PATH" "$FSD50K_PATH/FSD50K.doc.zip" \
  > "$LOG_PATH/fsd-doc.out" 2>&1 &

nohup unzip -d "$FSD50K_PATH" "$FSD50K_PATH/FSD50K.metadata.zip" \
  > "$LOG_PATH/fsd-meta.out" 2>&1 &


nohup zip -s 0 "$FSD50K_PATH/FSD50K.dev_audio.zip" -0 \
 --out "$FSD50K_PATH/FSD50K.dev_audio.full.zip" \
 > "$LOG_PATH/fsd-dev-merge.out" 2>&1 &

PID_FSD50K_DEV=$!

nohup zip -s 0 "$FSD50K_PATH/FSD50K.eval_audio.zip" -0 \
 --out "$FSD50K_PATH/FSD50K.eval_audio.full.zip" \
 > "$LOG_PATH/fsd-eval-merge.out" 2>&1 &

PID_FSD50K_EVAL=$!

echo "Waiting for eval zip files to merge ..."
wait $PID_FSD50K_EVAL
echo "Merge completed."
nohup unzip -d "$FSD50K_PATH" "$FSD50K_PATH/FSD50K.eval_audio.full.zip" \
 > "$LOG_PATH/fsd-eval.out" 2>&1 &

echo "Waiting for dev zip files to merge ..."
wait $PID_FSD50K_DEV
echo "Merge completed."
nohup unzip -d "$FSD50K_PATH" "$FSD50K_PATH/FSD50K.dev_audio.full.zip" \
 > "$LOG_PATH/fsd-dev.out" 2>&1 &
