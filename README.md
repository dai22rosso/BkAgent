# Triage Agent Demo: Safety-Aware Recovery Policies for Tool-Using Agents

## 方案C 完整版 Demo — 4-Action Triage Policy + Safety-Aware Reward

### Quick Start

```bash
# ===== Step 0: Verify environment =====
python -c "import torch, vllm, verl; print('OK')"

# ===== Step 1: Test environment + reward (no GPU needed) =====
python scripts/run_rollout.py --data data/train_demo.jsonl --mock --verbose

# ===== Step 2: Test with real model (needs GPU) =====
CUDA_VISIBLE_DEVICES=3,4,5 python scripts/run_rollout.py \
    --model ~/models/Qwen3-4B-Instruct \
    --data data/train_demo.jsonl \
    --max_episodes 5 --verbose

# ===== Step 3: Run rollout-only training loop (with real model) =====
CUDA_VISIBLE_DEVICES=3,4,5 python scripts/verl_train.py \
    --model ~/models/Qwen3-4B-Instruct \
    --data data/train_100.jsonl \
    --epochs 1 --alpha 0.5

# ===== Step 4: REAL verl GRPO training (gradient updates!) =====
# First, prepare parquet data:
python scripts/prepare_data.py --input data/train_100.jsonl --output data/train.parquet
python scripts/prepare_data.py --input data/train_demo.jsonl --output data/train_demo.parquet

# Then launch verl GRPO:
CUDA_VISIBLE_DEVICES=3,4,5 N_GPUS=3 bash scripts/run_grpo.sh

# ===== Step 5: Evaluate =====
python eval/evaluate.py --rollouts output/rollouts.jsonl --output output/eval_results.json
```

### What Each Script Does

| Script | GPU? | What it does |
|--------|------|-------------|
| `scripts/run_rollout.py --mock` | No | Tests env + reward with mock model |
| `scripts/run_rollout.py --model ...` | Yes | Runs real model inference, collects trajectories |
| `scripts/verl_train.py --demo` | No | Full training loop with mock (no gradient updates) |
| `scripts/verl_train.py --model ...` | Yes | Rollout collection + reward computation (no gradient updates) |
| `scripts/prepare_data.py` | No | Converts JSONL to parquet for verl |
| `scripts/run_grpo.sh` | Yes | **Real GRPO training via verl.trainer.main_ppo** |
| `eval/evaluate.py` | No | Computes all 9 metrics from rollout results |

### Architecture

```
scripts/run_rollout.py     -> Test inference + env (works)
scripts/verl_train.py      -> Rollout loop + reward (works, no gradient update)
scripts/run_grpo.sh         -> Real verl GRPO training with gradient updates
  calls: python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...
    uses: rewards/verl_reward.py::compute_score  (custom reward function)
    reads: data/train.parquet  (prepared by scripts/prepare_data.py)
```

### Reward Design (3-objective)

```
R = R_task + R_triage + alpha * R_safety + 0.3 * R_format

R_task:    +1.0 (correct answer)  |  0.0 (wrong)
R_triage:  +0.2 (correct action)  |  +0.15 (backtrack when needed)  |  +0.25 (replan when needed)
R_safety:  +0.5 (refuse harmful)  |  -0.3 (over-refuse)  |  -0.4 (miss harmful)
R_format:  +0.1 (valid tag)       |  -0.1 (invalid tag)

alpha = 0.5 (default), ablation: {0.0, 0.3, 0.5, 0.7, 1.0}
```

### Project Structure

```
triage_agent/
├── data/
│   ├── train_demo.jsonl         # 10 example episodes
│   ├── train_100.jsonl          # 100 generated episodes
│   ├── tools_registry.json      # 8 simulated tools
│   └── format_spec.md           # Data format spec
├── scripts/
│   ├── run_rollout.py           # Inference + trajectory collection
│   ├── verl_train.py            # Rollout training loop (no grad update)
│   ├── run_grpo.sh              # * Real verl GRPO launcher
│   ├── prepare_data.py          # JSONL to parquet conversion
│   └── generate_data.py         # Generate more training episodes
├── envs/
│   ├── triage_tool_env.py       # 4-action ToolEnv
│   ├── failure_injection.py     # 3-layer failure injection
│   └── user_simulator.py        # Simulated user for ask_clarify
├── rewards/
│   ├── verl_reward.py           # * verl-compatible compute_score()
│   ├── triage_reward.py         # Full 3-objective reward
│   └── safety_oracle.py         # Safety judgment
├── eval/
│   ├── evaluate.py              # 9-metric evaluation
│   ├── triage_analysis.py       # Triage decision matrix
│   └── pareto_plot.py           # Safety-Task Pareto curve
├── utils/
│   ├── action_parser.py         # Parse 6 action types
│   └── trajectory_utils.py      # State rollback
├── configs/
│   ├── train_config.yaml        # Hyperparameters
│   └── verl_grpo.yaml           # verl Hydra config (reference)
├── run_all.sh                   # Pipeline orchestrator
└── README.md
```

### Model
- Base: Qwen3-4B-Instruct (local: ~/models/Qwen3-4B-Instruct)
- Training: LoRA rank=64, GRPO with 8 rollouts per prompt
- Framework: verl 0.5.0 + vLLM 0.8.4
