"""
SFT Data Preparation: Merge JSONL files → verl SFT parquet

verl's SFT trainer (verl.trainer.sft_trainer) expects parquet with a 'messages' column.
Each row's 'messages' is a list of {"role": ..., "content": ...} dicts.

The SFT trainer is invoked with:
    torchrun -m verl.trainer.sft_trainer data.messages_key=messages ...

This script:
1. Reads all sft_agent_training_200*.jsonl files
2. Validates the data
3. Splits 90/10 into train/val
4. Saves as parquet

Usage:
    python scripts/prepare_sft_data.py \
        --input_dir data/ \
        --output_dir data/ \
        --val_ratio 0.1
"""

import argparse
import json
import os
import sys
import glob
import random
from collections import Counter

def load_all_jsonl(input_dir: str, pattern: str = "sft_agent_training_200*.jsonl") -> list:
    """Load all JSONL files matching the pattern."""
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print(f"ERROR: No files matching '{pattern}' found in {input_dir}")
        sys.exit(1)
    
    all_data = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line.strip()))
        print(f"  Loaded {fpath}: {sum(1 for _ in open(fpath))} lines")
    
    print(f"  Total: {len(all_data)} examples")
    return all_data


def validate(data: list) -> bool:
    """Quick validation."""
    issues = 0
    for i, d in enumerate(data):
        if "messages" not in d:
            print(f"  ERROR: Example {i} missing 'messages'")
            issues += 1
        msgs = d.get("messages", [])
        if not msgs:
            print(f"  ERROR: Example {i} has empty messages")
            issues += 1
        # Check roles alternate
        for j in range(1, len(msgs)):
            if msgs[j]["role"] == msgs[j-1]["role"] and msgs[j]["role"] != "system":
                print(f"  WARNING: Example {d.get('id', i)}: consecutive {msgs[j]['role']} at positions {j-1},{j}")
    
    if issues == 0:
        print("  Validation passed: no issues found")
    return issues == 0


def prepare_sft_parquet(
    input_dir: str,
    output_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Main function: load, validate, split, save."""
    import pandas as pd
    
    print("=" * 60)
    print("SFT Data Preparation for verl")
    print("=" * 60)
    
    # Load
    print("\n[1/4] Loading data...")
    all_data = load_all_jsonl(input_dir)
    
    # Validate
    print("\n[2/4] Validating...")
    validate(all_data)
    
    # Print distribution
    cats = Counter(d.get("category", "unknown") for d in all_data)
    print("\n  Category distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    
    # Split
    print(f"\n[3/4] Splitting (val_ratio={val_ratio}, seed={seed})...")
    random.seed(seed)
    
    # Stratified split by category
    by_cat = {}
    for d in all_data:
        cat = d.get("category", "unknown")
        by_cat.setdefault(cat, []).append(d)
    
    train_data = []
    val_data = []
    
    for cat, examples in by_cat.items():
        random.shuffle(examples)
        n_val = max(1, int(len(examples) * val_ratio))
        val_data.extend(examples[:n_val])
        train_data.extend(examples[n_val:])
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Convert to parquet
    # verl SFT expects: messages column (list of dicts)
    print("\n[4/4] Saving parquet...")
    
    def to_records(data_list):
        records = []
        for d in data_list:
            records.append({
                "messages": d["messages"],  # Keep as list of dicts — pandas handles this
            })
        return records
    
    train_records = to_records(train_data)
    val_records = to_records(val_data)
    
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "sft_train.parquet")
    val_path = os.path.join(output_dir, "sft_val.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"  Saved: {train_path} ({len(train_df)} rows)")
    print(f"  Saved: {val_path} ({len(val_df)} rows)")
    
    # Verify
    print("\n[Verify] Reading back parquet...")
    check_df = pd.read_parquet(train_path)
    print(f"  Columns: {list(check_df.columns)}")
    print(f"  Rows: {len(check_df)}")
    sample_msgs = check_df.iloc[0]["messages"]
    print(f"  Sample messages[0] type: {type(sample_msgs)}")
    if isinstance(sample_msgs, list):
        print(f"  Sample messages[0] roles: {[m['role'] for m in sample_msgs]}")
    
    # Also save the GRPO-format parquet for the post-SFT RL phase
    # This uses the corrected prepare_data.py format (dict, not json.dumps)
    print("\n[Bonus] Creating GRPO-format parquet from SFT data...")
    grpo_records = []
    for d in all_data:
        # Extract the user query from messages (skip system, take first user msg)
        user_msgs = [m for m in d["messages"] if m["role"] == "user"]
        query = user_msgs[0]["content"] if user_msgs else ""
        
        # For GRPO, we need prompt (system+user) and ground_truth
        system_msg = d["messages"][0] if d["messages"][0]["role"] == "system" else None
        
        prompt_msgs = []
        if system_msg:
            prompt_msgs.append(system_msg)
        prompt_msgs.append({"role": "user", "content": query})
        
        grpo_records.append({
            "prompt": prompt_msgs,  # list of dicts, not json.dumps
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({
                    "gold_answer": None,
                    "safety_label": "safe" if "refuse" not in d.get("category", "") else (
                        "harmful_explicit" if "explicit" in d.get("category", "") else
                        "harmful_implicit" if "implicit" in d.get("category", "") else "safe"
                    ),
                    "category": d.get("category", "normal"),
                    "expected_triage": _infer_triage(d),
                }),
            },
            "data_source": "triage_agent",
            "extra_info": {
                "episode_id": d.get("id", ""),
                "category": d.get("category", ""),
                "query": query,
            },
        })
    
    grpo_df = pd.DataFrame(grpo_records)
    grpo_train_path = os.path.join(output_dir, "grpo_train.parquet")
    grpo_df.to_parquet(grpo_train_path, index=False)
    print(f"  Saved GRPO format: {grpo_train_path} ({len(grpo_df)} rows)")
    
    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print(f"  SFT:  torchrun -m verl.trainer.sft_trainer data.train_files={train_path} data.val_files={val_path} ...")
    print(f"  GRPO: python -m verl.trainer.main_ppo data.train_files={grpo_train_path} ...")
    print("=" * 60)


def _infer_triage(d: dict) -> str:
    """Infer expected triage action from the SFT data."""
    cat = d.get("category", "")
    if "refuse" in cat or cat == "impossible":
        return "refuse"
    if cat == "backtrack":
        return "backtrack"
    if cat == "replan":
        return "replan"
    if cat == "ask_clarify":
        return "continue"
    return "continue"


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data for verl")
    parser.add_argument("--input_dir", type=str, default="data/",
                        help="Directory containing sft_agent_training_200*.jsonl files")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory for parquet files")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    prepare_sft_parquet(args.input_dir, args.output_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
