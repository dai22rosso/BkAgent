#!/bin/bash
# =============================================================================
# Triage Agent — verl GRPO Training Launcher
#
# This is the REAL training script that calls verl's main_ppo with GRPO config.
#
# Usage:
#   # Step 1: Prepare data (JSONL → parquet)
#   python scripts/prepare_data.py --input data/train_100.jsonl --output data/train.parquet
#
#   # Step 2: Launch training
#   bash scripts/run_grpo.sh
#
#   # Or with custom settings:
#   ALPHA=0.7 N_GPUS=4 bash scripts/run_grpo.sh
# =============================================================================

# set -x

# ---- Configurable variables ----
MODEL_PATH="${MODEL_PATH:-output/sft_triage/global_step_56_merged}"
TRAIN_DATA="${TRAIN_DATA:-data/grpo_train.parquet}"
VAL_DATA="${VAL_DATA:-data/grpo_val.parquet}"
REWARD_FN="${REWARD_FN:-rewards/verl_reward.py}"
N_GPUS="${N_GPUS:-3}"
ALPHA="${ALPHA:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/tangdianxing/dingyuandai_ckpt/output/grpo_triage_v3}"

# GRPO-specific
N_ROLLOUTS="${N_ROLLOUTS:-8}"           # Group size for GRPO
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LR="${LR:-1e-6}"

# LoRA
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA_PARAM="${LORA_ALPHA_PARAM:-128}"

echo "=================================================="
echo "Triage Agent — GRPO Training via verl"
echo "=================================================="
echo "Model:        $MODEL_PATH"
echo "Train data:   $TRAIN_DATA"
echo "Reward fn:    $REWARD_FN"
echo "GPUs:         $N_GPUS"
echo "Safety α:     $ALPHA (baked into reward function)"
echo "Rollouts/q:   $N_ROLLOUTS"
echo "Epochs:       $TOTAL_EPOCHS"
echo "LoRA rank:    $LORA_RANK"
echo "=================================================="

# ---- Verify data exists ----
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Run: python scripts/prepare_data.py --input data/train_100.jsonl --output $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$REWARD_FN" ]; then
    echo "ERROR: Reward function not found at $REWARD_FN"
    exit 1
fi

# ---- Fix Ray startup issues ----
# Disable Ray Dashboard (OpenTelemetry version conflict in some envs)
export RAY_DISABLE_DASHBOARD=1
# Increase Ray startup timeout
export RAY_BACKEND_LOG_LEVEL=warning

# ---- Launch verl GRPO training ----
# verl's entry point is main_ppo.py even for GRPO — GRPO is selected via algorithm.adv_estimator=grpo
export VLLM_USE_V1=0
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    custom_reward_function.path="$REWARD_FN" \
    custom_reward_function.name=compute_score \
    \
    reward_model.enable=False \
    \
    trainer.logger=['console','tensorboard'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.project_name=triage-agent \
    trainer.experiment_name="grpo_4action_alpha${ALPHA}" \
    trainer.default_local_dir="$OUTPUT_DIR"
