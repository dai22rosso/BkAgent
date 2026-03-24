"""
Data Verification Script

Reads the prepared parquet files and prints samples exactly as the model 
would see them during training. This lets you verify the data is correct
before committing to a training run.

Usage:
    python scripts/verify_data.py --data_dir data/
    python scripts/verify_data.py --data_dir data/ --n_samples 5 --show_grpo_only
"""

import argparse
import json
import os
import sys
import pandas as pd
from collections import Counter


def print_divider(title=""):
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")


def verify_sft(data_dir: str, n_samples: int = 3):
    """Verify SFT parquet files."""
    print_divider("SFT DATA VERIFICATION")
    
    for split in ["train", "val"]:
        path = os.path.join(data_dir, f"sft_{split}.parquet")
        if not os.path.exists(path):
            print(f"\n  WARNING: {path} not found!")
            continue
        
        df = pd.read_parquet(path)
        print(f"\n  [{split}] {path}: {len(df)} rows, columns={list(df.columns)}")
        
        # Stats
        msg_lens = []
        role_seqs = Counter()
        action_counts = Counter()
        
        for i, row in df.iterrows():
            msgs = row["messages"]
            msg_lens.append(len(msgs))
            role_seq = "→".join(m["role"][0] for m in msgs)  # s→u→a→u→a
            role_seqs[role_seq] += 1
            
            for msg in msgs:
                if msg["role"] == "assistant":
                    for tag in ["<tool_call>", "<final_answer>", "<recover>", "<refuse>"]:
                        if tag in msg.get("content", ""):
                            action_counts[tag] += 1
        
        print(f"\n  Message length distribution:")
        for length, count in sorted(Counter(msg_lens).items()):
            print(f"    {length} messages: {count}")
        
        print(f"\n  Role sequence patterns (top 5):")
        for seq, count in role_seqs.most_common(5):
            print(f"    {seq}: {count}")
        
        print(f"\n  Action tag counts:")
        for tag, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"    {tag}: {count}")
        
        # Print samples
        print(f"\n  --- Sample conversations ({split}) ---")
        indices = list(range(min(n_samples, len(df))))
        # Also pick one from the end to catch format differences
        if len(df) > n_samples:
            indices.append(len(df) - 1)
        
        for idx in indices:
            row = df.iloc[idx]
            msgs = row["messages"]
            print(f"\n  [Sample {idx}] ({len(msgs)} messages)")
            
            for j, msg in enumerate(msgs):
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    # Only show first 80 chars of system prompt
                    print(f"    [{role}] {content[:80]}...")
                elif role == "user":
                    print(f"    [{role}] {content[:200]}")
                elif role == "assistant":
                    print(f"    [{role}] {content[:200]}")
            print()


def verify_grpo(data_dir: str, n_samples: int = 3):
    """Verify GRPO parquet files."""
    print_divider("GRPO DATA VERIFICATION")
    
    for split in ["train", "val"]:
        path = os.path.join(data_dir, f"grpo_{split}.parquet")
        if not os.path.exists(path):
            print(f"\n  WARNING: {path} not found!")
            continue
        
        df = pd.read_parquet(path)
        print(f"\n  [{split}] {path}: {len(df)} rows, columns={list(df.columns)}")
        
        # Check prompt format
        prompt_lens = []
        prompt_types = Counter()
        safety_labels = Counter()
        expected_actions = Counter()
        
        for i, row in df.iterrows():
            prompt = row["prompt"]
            
            # Check type
            if isinstance(prompt, list):
                prompt_lens.append(len(prompt))
                prompt_types["list"] += 1
            elif hasattr(prompt, 'tolist'):  # numpy array from parquet
                prompt = prompt.tolist()
                prompt_lens.append(len(prompt))
                prompt_types["numpy_array"] += 1
            elif isinstance(prompt, str):
                prompt_types["string_BAD"] += 1
            else:
                prompt_types[f"other_{type(prompt).__name__}"] += 1
            
            # Check reward_model
            rm = row.get("reward_model", {})
            if isinstance(rm, dict):
                gt = rm.get("ground_truth", "{}")
                try:
                    gt_parsed = json.loads(gt) if isinstance(gt, str) else gt
                    safety_labels[gt_parsed.get("safety_label", "MISSING")] += 1
                    expected_actions[gt_parsed.get("expected_triage", "MISSING")] += 1
                except:
                    safety_labels["PARSE_ERROR"] += 1
        
        print(f"\n  Prompt type distribution:")
        for ptype, count in prompt_types.items():
            print(f"    {ptype}: {count}")
        
        print(f"\n  Prompt length distribution (number of messages):")
        for length, count in sorted(Counter(prompt_lens).items()):
            print(f"    {length} messages: {count}")
        
        print(f"\n  Safety label distribution:")
        for label, count in sorted(safety_labels.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count}")
        
        print(f"\n  Expected action distribution:")
        for act, count in sorted(expected_actions.items(), key=lambda x: -x[1]):
            print(f"    {act}: {count}")
        
        # Print samples — show exactly what tokenizer will see
        print(f"\n  --- Sample prompts ({split}) ---")
        
        # Pick diverse samples: beginning, middle, end
        sample_indices = [0]
        if len(df) > 3:
            sample_indices.append(len(df) // 2)
        if len(df) > 1:
            sample_indices.append(len(df) - 1)
        sample_indices = sample_indices[:n_samples + 1]
        
        for idx in sample_indices:
            row = df.iloc[idx]
            prompt = row["prompt"]
            if hasattr(prompt, 'tolist'):
                prompt = prompt.tolist()
            
            rm = row.get("reward_model", {})
            ei = row.get("extra_info", {})
            
            # Parse ground truth
            gt_str = rm.get("ground_truth", "{}") if isinstance(rm, dict) else "{}"
            try:
                gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
            except:
                gt = {"error": "could not parse"}
            
            print(f"\n  [Sample {idx}] prompt has {len(prompt)} messages")
            print(f"    category: {ei.get('category', 'N/A') if isinstance(ei, dict) else 'N/A'}")
            print(f"    safety: {gt.get('safety_label', 'N/A')}")
            print(f"    expected: {gt.get('expected_triage', 'N/A')}")
            
            for j, msg in enumerate(prompt):
                if isinstance(msg, dict):
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                else:
                    role = "?"
                    content = str(msg)[:100]
                
                if role == "system":
                    print(f"    [{role}] (system prompt, {len(content)} chars)")
                else:
                    print(f"    [{role}] {content[:200]}")
            
            print()


def verify_consistency(data_dir: str):
    """Cross-check SFT and GRPO data for consistency."""
    print_divider("CONSISTENCY CHECKS")
    
    # Check system prompt is the same
    sft_path = os.path.join(data_dir, "sft_train.parquet")
    grpo_path = os.path.join(data_dir, "grpo_train.parquet")
    
    if not os.path.exists(sft_path) or not os.path.exists(grpo_path):
        print("  Cannot run consistency checks - files missing")
        return
    
    sft_df = pd.read_parquet(sft_path)
    grpo_df = pd.read_parquet(grpo_path)
    
    # Check system prompts match
    sft_system = sft_df.iloc[0]["messages"][0]["content"]
    
    grpo_prompt = grpo_df.iloc[0]["prompt"]
    if hasattr(grpo_prompt, 'tolist'):
        grpo_prompt = grpo_prompt.tolist()
    grpo_system = grpo_prompt[0]["content"] if isinstance(grpo_prompt[0], dict) else "N/A"
    
    if sft_system == grpo_system:
        print("  ✓ System prompts match between SFT and GRPO")
    else:
        print("  ✗ System prompts DIFFER!")
        print(f"    SFT:  {sft_system[:80]}...")
        print(f"    GRPO: {grpo_system[:80]}...")
    
    # Check no overlap between SFT queries and GRPO queries
    sft_queries = set()
    for _, row in sft_df.iterrows():
        msgs = row["messages"]
        for m in msgs:
            if m["role"] == "user" and "<tool_result>" not in m.get("content", ""):
                sft_queries.add(m["content"][:100])
                break
    
    grpo_queries = set()
    for _, row in grpo_df.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user" and "<tool_result>" not in m.get("content", ""):
                grpo_queries.add(m["content"][:100])
                break
    
    overlap = sft_queries & grpo_queries
    print(f"\n  SFT unique queries: {len(sft_queries)}")
    print(f"  GRPO unique queries: {len(grpo_queries)}")
    print(f"  Overlap: {len(overlap)}")
    if overlap:
        print(f"  ✗ WARNING: {len(overlap)} overlapping queries!")
        for q in list(overlap)[:3]:
            print(f"    '{q}'")
    else:
        print("  ✓ No query overlap between SFT and GRPO")
    
    # Check GRPO prompt is NOT a json.dumps string (the old bug)
    bad_prompts = 0
    for _, row in grpo_df.head(10).iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):
            bad_prompts += 1
    
    if bad_prompts > 0:
        print(f"\n  ✗ CRITICAL: {bad_prompts}/10 GRPO prompts are strings (json.dumps bug!)")
    else:
        print(f"\n  ✓ GRPO prompts are list/array (correct format)")
    
    # File size sanity check
    sft_size = os.path.getsize(sft_path) / 1024 / 1024
    grpo_size = os.path.getsize(grpo_path) / 1024 / 1024
    print(f"\n  File sizes: SFT={sft_size:.2f}MB, GRPO={grpo_size:.2f}MB")
    
    if grpo_size < 0.1:
        print("  ✗ WARNING: GRPO file suspiciously small!")
    else:
        print("  ✓ File sizes look reasonable")


def main():
    parser = argparse.ArgumentParser(description="Verify prepared data")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--show_sft_only", action="store_true")
    parser.add_argument("--show_grpo_only", action="store_true")
    args = parser.parse_args()
    
    if not args.show_grpo_only:
        verify_sft(args.data_dir, args.n_samples)
    
    if not args.show_sft_only:
        verify_grpo(args.data_dir, args.n_samples)
    
    verify_consistency(args.data_dir)
    
    print_divider("VERIFICATION COMPLETE")
    print("  If everything looks good, proceed with training.")
    print("  If not, fix the source JSONL files and re-run prepare_data_v2.py")


if __name__ == "__main__":
    main()
