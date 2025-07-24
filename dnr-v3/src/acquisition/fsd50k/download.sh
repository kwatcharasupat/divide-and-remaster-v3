#!/bin/bash

set -e

FSD50K_ZENODO_ID="4060432"

FSD50K_PATH="$RAW_DATA_ROOT/fsd50k"
LOG_PATH="$FSD50K_PATH/_log"

mkdir -p $FSD50K_PATH
mkdir -p $LOG_PATH

cd $FSD50K_PATH

zenodo_get -w "$LOG_PATH/_urls.txt" $FSD50K_ZENODO_ID

while read -r URL; do
    echo "Downloading $URL"
    SHORTNAME=$(basename "$URL")
    nohup wget -b -P "$FSD50K_ZIP_PATH" "$URL" -o "$LOG_PATH/_$SHORTNAME.log" &> "$LOG_PATH/nohup.out" &
done < "$LOG_PATH/_urls.txt"

watch -d "ps aux | grep wget"
