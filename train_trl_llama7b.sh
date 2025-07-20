#!/bin/bash

# Fine-tune LLaMA-7B using TRL's SFTTrainer
# Based on Stanford Alpaca training configuration

# Set your paths here
MODEL_PATH="/root/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9" # This will auto-download if you have access
OUTPUT_DIR="outputs"  # Replace with your output directory
DATA_PATH="./alpaca_data.json"
MASTER_PORT=29500  # Change if needed

# Run training with 4 GPUs
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train_trl.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 512 \
    --report_to none
