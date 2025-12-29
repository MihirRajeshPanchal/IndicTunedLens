#!/bin/bash

# Set Hugging Face cache directories
export HF_HOME="/mnt/storage/deeksha/.cache/huggingface"
export HF_DATASETS_CACHE="/mnt/storage/deeksha/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/mnt/storage/deeksha/.cache/huggingface/transformers"

# Define all languages
languages=("hi" "bn" "en" "gu" "kn" "ml" "mr" "ne" "ta" "te")

MODEL_NAME="sarvamai/sarvam-1"
BASE_DATA_PATH="/mnt/storage/deeksha/indictunedlens/data"
BASE_RESULTS_DIR="/mnt/storage/deeksha/indictunedlens/results"
BASE_OUTPUT_DIR="/mnt/storage/deeksha/indictunedlens/results_with_matches"
LOG_DIR="indictunedlens/logs"

echo "Starting gold match for all languages..."

# Run gold match for each language
for lang in "${languages[@]}"; do
    echo "Running gold match for language: $lang"
    
    python indictunedlens/gold_match.py \
        --original_csv "$BASE_DATA_PATH/m_mmlu_${lang}.csv" \
        --results_dir "$BASE_RESULTS_DIR/m_mmlu_${lang}" \
        --output_dir "$BASE_OUTPUT_DIR/m_mmlu_${lang}" \
        --model_name "$MODEL_NAME" \
        --lens_type both \
        > "$LOG_DIR/gold_match_m_mmlu_${lang}.log" 2>&1 &
done

# Wait for all background processes to complete
echo "Waiting for all gold match jobs to complete..."
wait

echo "All gold match jobs completed!"