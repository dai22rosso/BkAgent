#!/bin/bash
# =============================================================================
# Automated GRPO Pipeline: Train (A,B,C) -> Sequential Merge & Eval -> Cleanup
# =============================================================================
set -e

# ---- Config ----
BASE_MODEL="$HOME/models/Qwen3-4B-Instruct"
SFT_MODEL="output/sft_triage/global_step_56_merged"
TEST_DATA="data/test_300.jsonl"
CKPT_ROOT="/data/tangdianxing/dingyuandai_ckpt/output"

mkdir -p ./output
COMPARISON_LOG="$(pwd)/output/comparison.log"
echo "================ EVALUATION COMPARISON LOG ================" > "$COMPARISON_LOG"
echo "Pipeline Started at: $(date)" >> "$COMPARISON_LOG"

# ---- Step 0: Eval Base + SFT once (they don't change across variants) ----
echo "=================================================="
echo "📊 Evaluating E1 (Base) + E2 (SFT) baseline once..."
echo "=================================================="
echo -e "\n\n================ BASELINE: E1 (Base) + E2 (SFT) ================" >> "$COMPARISON_LOG"

CUDA_VISIBLE_DEVICES=0 python scripts/eval_models.py \
    --e1_model "$BASE_MODEL" \
    --e2_model "$SFT_MODEL" \
    --e4_model "$SFT_MODEL" \
    --test_data "$TEST_DATA" \
    --output "output/eval_baseline.json" >> "$COMPARISON_LOG" 2>&1

echo "✅ Baseline evaluation done."

# ---- Loop over variants ----
for VARIANT in A B C; do
    echo ""
    echo "=================================================="
    echo "🚀 STARTING PIPELINE FOR REWARD VARIANT: $VARIANT"
    echo "=================================================="

    # Set env vars for this variant
    export REWARD_VARIANT=$VARIANT
    export OUTPUT_DIR="${CKPT_ROOT}/grpo_v3_${VARIANT}"
    TRAIN_LOG="./output/grpo_variant_${VARIANT}.log"

    # ---- 1. Train ----
    echo "🧠 Training Variant $VARIANT..."
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    N_GPUS=8 \
    REWARD_VARIANT=$VARIANT \
    OUTPUT_DIR="$OUTPUT_DIR" \
    TOTAL_EPOCHS=1 \
    bash scripts/run_grpo.sh 2>&1 | tee "$TRAIN_LOG"

    # ---- 2. Find checkpoints ----
    STEP_DIRS=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name "global_step_*" ! -name "*_merged" | sort -V)

    if [ -z "$STEP_DIRS" ]; then
        echo "❌ ERROR: No checkpoints found in $OUTPUT_DIR. Skipping."
        continue
    fi

    echo "📁 Found checkpoints: $(echo $STEP_DIRS | tr '\n' ' ')"

    # ---- 3. Merge + Eval each checkpoint ----
    for STEP_DIR in $STEP_DIRS; do
        STEP_NAME=$(basename "$STEP_DIR")
        MERGED_DIR="${STEP_DIR}_merged"

        echo "--------------------------------------------------"
        echo "🎯 Variant $VARIANT | $STEP_NAME"

        # [A] Merge
        echo "   🔄 Merging..."
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${STEP_DIR}/actor" \
            --target_dir "$MERGED_DIR"

        # [B] Delete unmerged shards
        echo "   🗑️ Deleting unmerged shards..."
        rm -rf "${STEP_DIR}"

        # [C] Eval (skip E1 — already done above; only compare SFT vs GRPO)
        EVAL_OUTPUT="output/eval_${VARIANT}_${STEP_NAME}.json"
        echo "   📊 Evaluating..."
        echo -e "\n\n================ VARIANT $VARIANT | $STEP_NAME ================" >> "$COMPARISON_LOG"
        echo "Model: $MERGED_DIR" >> "$COMPARISON_LOG"

        CUDA_VISIBLE_DEVICES=0 python scripts/eval_models.py \
            --e1_model "$BASE_MODEL" \
            --e2_model "$SFT_MODEL" \
            --e4_model "$MERGED_DIR" \
            --test_data "$TEST_DATA" \
            --output "$EVAL_OUTPUT" >> "$COMPARISON_LOG" 2>&1

        echo "   ✅ Done: $STEP_NAME"
    done

    echo "✅ Variant $VARIANT complete!"
done

echo ""
echo "=================================================="
echo "🎉 All variants (A, B, C) trained + evaluated!"
echo "Results: $COMPARISON_LOG"
echo "=================================================="
echo "Pipeline Finished at: $(date)" >> "$COMPARISON_LOG"
