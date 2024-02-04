#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python app.py \
    --model-name openai/whisper-large-v3 \
    --batch-size 24 \
    --port 9000
