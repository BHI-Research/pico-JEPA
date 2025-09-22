#!/bin/bash

# This script runs inference on all models for all classes defined in classInd.txt, processing all .mp4 files
# within their corresponding directories.
# Use: ./infers_part.sh

CLASS_NAMES_FILE="videos/classInd.txt"
PROJECT_ROOT=$(pwd)
MAX_MODEL=`ls -1 $PROJECT_ROOT/pico_jepa_classifier*[1-9].pth | wc -l`
if [ ! -f "${PROJECT_ROOT}/${CLASS_NAMES_FILE}" ]; then
    echo "Error: Class name file not found: ${PROJECT_ROOT}/${CLASS_NAMES_FILE}"
    exit 1
fi

while IFS= read -r line; do
    
    CLASS_NAME=$(echo "$line" | sed 's/^[0-9]\+ //g' | tr -d '\r')
    
    if [ -z "$CLASS_NAME" ]; then
        continue
    fi

    echo "=================================================="
    echo "==== Processing class: $CLASS_NAME ===="
    echo "=================================================="

    OUTPUT_FILE="${CLASS_NAME}.txt"
    VIDEO_DIR="${PROJECT_ROOT}/videos/infer/${CLASS_NAME}"
    
    if [ ! -d "$VIDEO_DIR" ]; then
        echo "Error: Video directory not found for class '$CLASS_NAME': $VIDEO_DIR"
        continue 
    fi
    
    > "$OUTPUT_FILE"  

    for video_file in "$VIDEO_DIR"/*.mp4; do
        
        if [ ! -e "$video_file" ]; then
            echo "No .mp4 files were found in the '$VIDEO_DIR' directory. Exiting the video loop."
            break
        fi

        echo "---- Processing video: $(basename "$video_file") ----"

	for model_num in $(seq 1 $MAX_MODEL); do

            echo "--- Model $model_num ---"

            python app/infer_video.py \
                    --video_path "$video_file" \
                    --model_path ${PROJECT_ROOT}/pico_jepa_classifier${model_num}.pth \
                    --config_path ${PROJECT_ROOT}/configs/config-${model_num}.yaml \
                    --class_names_file "${PROJECT_ROOT}/${CLASS_NAMES_FILE}" |  tee -a "${PROJECT_ROOT}/$OUTPUT_FILE"
        done
    done
done < "${PROJECT_ROOT}/${CLASS_NAMES_FILE}"

echo "=== Inference process completed for all classes ==="

