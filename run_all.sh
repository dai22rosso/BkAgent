#!/bin/bash
# =============================================================================
# Triage Agent — Full Pipeline Runner
# =============================================================================
# Usage:
#   bash run_all.sh demo          # Quick demo (mock model, ~30 seconds)
#   bash run_all.sh train         # Full training (requires GPU)
#   bash run_all.sh ablation      # Alpha ablation study
#   bash run_all.sh eval_only     # Evaluate existing checkpoints
# =============================================================================

set -e
cd "$(dirname "$0")"

MODE="${1:-demo}"
MODEL="${MODEL:-~/models/Qwen3-4B-Instruct}"
DATA_SMALL="data/train_demo.jsonl"
DATA_FULL="data/train_100.jsonl"
OUTPUT="output"

echo "=================================================="
echo "Triage Agent Pipeline — Mode: $MODE"
echo "=================================================="

case "$MODE" in
    demo)
        echo ""
        echo "[1/4] Generating test data (100 episodes)..."
        python scripts/generate_data.py --num_episodes 100 --output "$DATA_FULL" --seed 42
        
        echo ""
        echo "[2/4] Running demo rollouts (10 episodes, mock model)..."
        python scripts/run_rollout.py \
            --data "$DATA_SMALL" \
            --mock --verbose \
            --output "$OUTPUT/demo_rollouts.jsonl"
        
        echo ""
        echo "[3/4] Running evaluation..."
        python eval/evaluate.py \
            --rollouts "$OUTPUT/demo_rollouts.jsonl" \
            --output "$OUTPUT/eval_results.json"
        
        echo ""
        echo "[4/4] Running triage analysis..."
        python eval/triage_analysis.py \
            --rollouts "$OUTPUT/demo_rollouts.jsonl" \
            --output "$OUTPUT/triage_analysis.json"
        
        echo ""
        echo "Demo complete! Check output/ for results."
        ;;
    
    train)
        echo ""
        echo "[1/5] Generating training data (100 episodes)..."
        python scripts/generate_data.py --num_episodes 100 --output "$DATA_FULL" --seed 42
        
        echo ""
        echo "[2/5] Running baseline rollouts (before training)..."
        python scripts/run_rollout.py \
            --data "$DATA_FULL" \
            --model "$MODEL" \
            --max_episodes 20 \
            --output "$OUTPUT/baseline_rollouts.jsonl"
        
        echo ""
        echo "[3/5] Training with GRPO (α=0.5)..."
        python scripts/verl_train.py \
            --model "$MODEL" \
            --data "$DATA_FULL" \
            --output "$OUTPUT/grpo_alpha_05" \
            --epochs 3 \
            --batch_size 4 \
            --rollouts_per_prompt 8 \
            --alpha 0.5
        
        echo ""
        echo "[4/5] Evaluating trained model..."
        python eval/evaluate.py \
            --data "$DATA_FULL" \
            --model "$MODEL" \
            --output "$OUTPUT/eval_trained.json"
        
        echo ""
        echo "[5/5] Triage analysis..."
        python eval/triage_analysis.py \
            --rollouts "$OUTPUT/grpo_alpha_05/rollouts.jsonl" \
            --output "$OUTPUT/triage_trained.json" 2>/dev/null || echo "Skipped (no rollout file)"
        
        echo ""
        echo "Training complete!"
        ;;
    
    ablation)
        echo ""
        echo "Running alpha ablation: α ∈ {0.0, 0.3, 0.5, 0.7, 1.0}"
        echo ""
        
        for alpha in 0.0 0.3 0.5 0.7 1.0; do
            echo "--- Training with α=$alpha ---"
            python scripts/verl_train.py \
                --model "$MODEL" \
                --data "$DATA_FULL" \
                --output "$OUTPUT/grpo_alpha_${alpha//./_}" \
                --epochs 3 \
                --alpha "$alpha" \
                --demo  # Remove --demo for real training
            
            echo "--- Evaluating α=$alpha ---"
            python eval/evaluate.py \
                --mock \
                --data "$DATA_FULL" \
                --output "$OUTPUT/eval_alpha_${alpha//./_}.json"
        done
        
        echo ""
        echo "Generating Pareto curve..."
        python eval/pareto_plot.py \
            --results \
                "$OUTPUT/eval_alpha_0_0.json" \
                "$OUTPUT/eval_alpha_0_3.json" \
                "$OUTPUT/eval_alpha_0_5.json" \
                "$OUTPUT/eval_alpha_0_7.json" \
                "$OUTPUT/eval_alpha_1_0.json" \
            --alphas 0.0 0.3 0.5 0.7 1.0 \
            --output "$OUTPUT/pareto_curve.png"
        
        echo ""
        echo "Ablation complete! Check output/pareto_curve.png"
        ;;
    
    eval_only)
        echo ""
        echo "Evaluating from existing rollouts..."
        if [ -f "$OUTPUT/demo_rollouts.jsonl" ]; then
            python eval/evaluate.py \
                --rollouts "$OUTPUT/demo_rollouts.jsonl" \
                --output "$OUTPUT/eval_results.json"
            python eval/triage_analysis.py \
                --rollouts "$OUTPUT/demo_rollouts.jsonl" \
                --output "$OUTPUT/triage_analysis.json"
        else
            echo "No rollouts found. Run 'bash run_all.sh demo' first."
        fi
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash run_all.sh {demo|train|ablation|eval_only}"
        exit 1
        ;;
esac
