#!/bin/bash

# Fine-tune LLaMA-13B using TRL's SFTTrainer
# Based on Stanford Alpaca training configuration

# Set your paths here
MODEL_PATH="<your_path_to_hf_converted_llama_13b_ckpt_and_tokenizer>"  # Replace with your LLaMA-13B path
OUTPUT_DIR="<your_output_dir>"  # Replace with your output directory
DATA_PATH="./alpaca_data.json"
MASTER_PORT=29500  # Change if needed

# Run training with 4 GPUs
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train_trl.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 512