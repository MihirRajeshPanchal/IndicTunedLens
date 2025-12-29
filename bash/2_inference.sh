#!/bin/bash

# Set Hugging Face cache directories
export HF_HOME="/mnt/storage/deeksha/.cache/huggingface"
export HF_DATASETS_CACHE="/mnt/storage/deeksha/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/mnt/storage/deeksha/.cache/huggingface/transformers"

# Define languages and their corresponding devices
declare -A lang_device=(
    ["bn"]="cuda:0"
    ["en"]="cuda:1"
    ["gu"]="cuda:2"
    ["hi"]="cuda:3"
    ["kn"]="cuda:0"
    ["ml"]="cuda:3"
    ["mr"]="cuda:4"
    ["ne"]="cuda:5"
    ["ta"]="cuda:2"
    ["te"]="cuda:1"
)

MODEL_NAME="sarvamai/sarvam-1"
LENS_PATH="/mnt/storage/deeksha/indictunedlens/trained_lens/sarvamai/sarvam-1"
BASE_DATA_PATH="/mnt/storage/deeksha/indictunedlens/data"
BASE_OUTPUT_DIR="indictunedlens/results"
LOG_DIR="indictunedlens/logs"

echo "Starting inference for all languages..."

# Run inference for each language
for lang in "${!lang_device[@]}"; do
    device="${lang_device[$lang]}"
    
    echo "Running inference for language: $lang on device: $device"
    
    python indictunedlens/inference.py \
        --model_name "$MODEL_NAME" \
        --lens_path "$LENS_PATH" \
        --csv_path "$BASE_DATA_PATH/m_mmlu_${lang}.csv" \
        --output_dir "$BASE_OUTPUT_DIR/m_mmlu_${lang}" \
        --k 10 \
        --head -1 \
        --device "$device" \
        > "$LOG_DIR/inference_sarvamai_sarvam-1_m_mmlu_${lang}.log" 2>&1 &
done

# Wait for all background processes to complete
echo "Waiting for all inference jobs to complete..."
wait

echo "All inference jobs completed!"