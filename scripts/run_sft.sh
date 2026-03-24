#!/bin/bash
# =============================================================================
# Triage Agent — SFT Training via verl
#
# This trains the base model to learn the action tag format before GRPO.
# Uses verl's built-in SFT trainer (torchrun -m verl.trainer.sft_trainer).
#
# Usage:
#   # Step 1: Prepare data
#   python scripts/prepare_sft_data.py --input_dir data/ --output_dir data/
#
#   # Step 2: Run SFT
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 N_GPUS=8 bash scripts/run_sft.sh
#
#   # Step 3: Then run GRPO on the SFT checkpoint
#   MODEL_PATH=output/sft_triage/global_step_XXX/huggingface bash scripts/run_grpo.sh
# =============================================================================

set -x

# ---- Configurable variables ----
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-4B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-data/sft_train.parquet}"
VAL_DATA="${VAL_DATA:-data/sft_val.parquet}"
N_GPUS="${N_GPUS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/tangdianxing/dingyuandai_ckpt/output/sft_triage_e2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
LR="${LR:-2e-5}"
MICRO_BATCH="${MICRO_BATCH:-2}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"

echo "=================================================="
echo "Triage Agent — SFT Training via verl"
echo "=================================================="
echo "Model:        $MODEL_PATH"
echo "Train data:   $TRAIN_DATA"
echo "Val data:     $VAL_DATA"
echo "GPUs:         $N_GPUS"
echo "Epochs:       $TOTAL_EPOCHS"
echo "LR:           $LR"
echo "Micro batch:  $MICRO_BATCH"
echo "=================================================="

# Verify data
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Run: python scripts/prepare_sft_data.py --input_dir data/ --output_dir data/"
    exit 1
fi

# Launch verl SFT trainer
# 使用正确的 fsdp_sft_trainer，并适配多轮对话和模型加载参数
PYTHONUNBUFFERED=1 torchrun \
    --nproc_per_node=$N_GPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=$TRAIN_BATCH \
    data.max_length=4096 \
    model.partial_pretrain="$MODEL_PATH" \
    model.fsdp_config.model_dtype=bfloat16 \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    optim.lr=$LR \
    optim.weight_decay=0.01 \
    trainer.project_name=triage-agent-sft \
    trainer.experiment_name=sft_action_tags \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.logger=console
