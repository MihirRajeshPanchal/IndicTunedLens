#!/bin/bash

# Set Hugging Face cache directories
export HF_HOME="/mnt/storage/deeksha/.cache/huggingface"
export HF_DATASETS_CACHE="/mnt/storage/deeksha/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/mnt/storage/deeksha/.cache/huggingface/transformers"

# Train the tuned lens
echo "Starting tuned lens training..."
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    -m tuned_lens train \
    sarvamai/sarvam-1 \
    /mnt/storage/deeksha/indictunedlens/sangraha.jsonl \
    --per-gpu-batch-size 1 \
    -o /mnt/storage/deeksha/indictunedlens/trained_lens/sarvamai/sarvam-1 \
    --fsdp \
    >> indictunedlens/logs/sarvamai_sarvam-1.log 2>&1

echo "Training completed!"

# Download dataset
echo "Downloading dataset..."
python indictunedlens/download_dataset.py

echo "Dataset download completed!"