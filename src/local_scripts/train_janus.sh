#!/bin/bash

export NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# 设置 GPU 列表
GPUS="0"

# 设置 Weights & Biases (WandB) 配置
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=vision-reasoning
export WANDB_API_KEY="dd6af3f5f51014cd49b6f4ec4e4943d917b46ed6"
export WANDB_RUN_NAME=Janus-Pro-2B-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
wandb login $WANDB_API_KEY

# cd /home/tiger/multimodal-open-r1
# pip3 install -e ".[dev]"
# pip3 install wandb==0.18.3

torchrun --nproc_per_node=1 \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path deepseek-ai/Janus-Pro-1B \
    --dataset_name lmms-lab/multimodal-open-r1-8k-verified \
    --max_prompt_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2359296 \
    --save_total_limit 8 \
    --num_train_epochs 1 \
    --run_name $WANDB_RUN_NAME