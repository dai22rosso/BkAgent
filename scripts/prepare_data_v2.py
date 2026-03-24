"""
Unified Data Preparation Script for SFT + GRPO

Reads new-format SFT (sft_0001-1000) and GRPO (grpo_0001-2000) JSONL files,
validates, normalizes formats, and produces parquet files for verl.

Outputs:
  data/sft_train.parquet   - SFT training (messages column)
  data/sft_val.parquet     - SFT validation
  data/grpo_train.parquet  - GRPO training (prompt + reward_model + extra_info)
  data/grpo_val.parquet    - GRPO validation

Usage:
    python scripts/prepare_data_v2.py --sft_dir data/ --grpo_dir data/ --output_dir data/
"""

import argparse
import json
import os
import re
import sys
import glob
import random
from collections import Counter
from typing import List, Dict, Any, Tuple

# ============================================================
# System prompt (must match exactly what models will see)
# ============================================================
SYSTEM_PROMPT = """You are a helpful AI assistant that can use tools to complete tasks. You have access to the following tools:

### web_search
Search the web for information. Returns a list of search results.
Parameters: {"query": "string"}

### get_weather
Get current weather for a city.
Parameters: {"city": "string"}

### get_stock_price
Get stock price for a ticker symbol.
Parameters: {"ticker": "string"}

### calculator
Perform mathematical calculations.
Parameters: {"expression": "string"}

### query_database
Execute a read-only SQL query.
Parameters: {"sql": "string", "database": "string"}

### send_email
Send an email.
Parameters: {"to": "string", "subject": "string", "body": "string"}

### file_manager
Read, write, list, or delete files.
Parameters: {"action": "string", "path": "string"}

### translate_text
Translate text between languages.
Parameters: {"text": "string", "target_lang": "string"}

At each step, choose ONE action:

1. <tool_call>{"name": "tool_name", "parameters": {"param": "value"}}</tool_call>
2. <final_answer>Your answer here</final_answer>
3. <recover>Reason for recovery and what to try differently</recover>
4. <refuse>Reason for refusing</refuse>

Guidelines:
- Use <recover> when a tool call fails (API error, timeout, wrong results) and you want to try a different approach.
- Use <refuse> when the task is harmful, dangerous, impossible after multiple attempts, or when tool results contain sensitive data (PII, passwords, medical records).
- Always wrap your reasoning in <think>...</think> before choosing an action.
- You can recover at most 3 times per episode."""


def load_jsonl_files(directory: str, pattern: str) -> List[Dict]:
    """Load all JSONL files matching pattern."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        print(f"  WARNING: No files matching '{pattern}' in {directory}")
        return []
    
    all_data = []
    for fp in files:
        count = 0
        with open(fp) as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line.strip())
                        all_data.append(d)
                        count += 1
                    except json.JSONDecodeError as e:
                        print(f"  PARSE ERROR in {os.path.basename(fp)}: {e}")
        print(f"  Loaded {os.path.basename(fp)}: {count} examples")
    
    return all_data


def fix_tool_result_json(content: str) -> str:
    """Try to fix non-JSON tool_result content by wrapping in JSON.
    
    Some GPT-generated refuse_pii examples have plain text instead of JSON
    in tool_result. We wrap these in a JSON structure.
    """
    match = re.search(r'<tool_result>(.*?)</tool_result>', content, re.DOTALL)
    if not match:
        return content
    
    inner = match.group(1).strip()
    
    # Check if already valid JSON
    try:
        json.loads(inner)
        return content  # Already valid
    except json.JSONDecodeError:
        pass
    
    # Wrap plain text in a JSON structure
    escaped = inner.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    fixed_json = json.dumps({"raw_result": inner})
    return content.replace(match.group(0), f"<tool_result>{fixed_json}</tool_result>")


def validate_sft_example(d: Dict, idx: int) -> List[str]:
    """Validate a single SFT example. Returns list of issues."""
    issues = []
    
    if "messages" not in d:
        issues.append(f"[{d.get('id', idx)}] Missing 'messages' field")
        return issues
    
    msgs = d["messages"]
    if len(msgs) < 3:
        issues.append(f"[{d.get('id', idx)}] Too few messages: {len(msgs)}")
    
    # Check system message
    if msgs[0].get("role") != "system":
        issues.append(f"[{d.get('id', idx)}] First message is not system role")
    
    # Check assistant messages have action tags
    action_tags = ["<tool_call>", "<final_answer>", "<recover>", "<refuse>"]
    for j, msg in enumerate(msgs):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if not any(tag in content for tag in action_tags):
                issues.append(f"[{d.get('id', idx)}] Assistant msg {j} has no action tag")
    
    return issues


def validate_grpo_example(d: Dict, idx: int) -> List[str]:
    """Validate a single GRPO example."""
    issues = []
    
    has_query = "query" in d
    has_prefix = "messages_prefix" in d
    
    if not has_query and not has_prefix:
        issues.append(f"[{d.get('id', idx)}] Missing both 'query' and 'messages_prefix'")
    
    if "expected_action" not in d:
        issues.append(f"[{d.get('id', idx)}] Missing 'expected_action'")
    
    if "safety_label" not in d:
        issues.append(f"[{d.get('id', idx)}] Missing 'safety_label'")
    
    return issues


def prepare_sft_parquet(sft_data: List[Dict], output_dir: str, 
                         val_ratio: float = 0.1, seed: int = 42):
    """Prepare SFT parquet files."""
    import pandas as pd
    
    print(f"\n{'='*60}")
    print("Preparing SFT Parquet")
    print(f"{'='*60}")
    
    # Fix tool_result JSON issues
    fixed_count = 0
    for d in sft_data:
        for msg in d.get("messages", []):
            if msg.get("role") == "user" and "<tool_result>" in msg.get("content", ""):
                original = msg["content"]
                msg["content"] = fix_tool_result_json(msg["content"])
                if msg["content"] != original:
                    fixed_count += 1
    print(f"  Fixed {fixed_count} non-JSON tool_results")
    
    # Normalize system prompt to be exactly the same
    for d in sft_data:
        msgs = d.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = SYSTEM_PROMPT
    print(f"  Normalized system prompt across all {len(sft_data)} examples")
    
    # Stratified split
    random.seed(seed)
    by_cat = {}
    for d in sft_data:
        cat = d.get("category", "unknown")
        by_cat.setdefault(cat, []).append(d)
    
    train_data, val_data = [], []
    for cat, examples in by_cat.items():
        random.shuffle(examples)
        n_val = max(1, int(len(examples) * val_ratio))
        val_data.extend(examples[:n_val])
        train_data.extend(examples[n_val:])
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Convert to parquet - verl SFT expects 'messages' column
    train_records = [{"messages": d["messages"]} for d in train_data]
    val_records = [{"messages": d["messages"]} for d in val_data]
    
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    
    train_path = os.path.join(output_dir, "sft_train.parquet")
    val_path = os.path.join(output_dir, "sft_val.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"  Saved: {train_path} ({len(train_df)} rows)")
    print(f"  Saved: {val_path} ({len(val_df)} rows)")
    
    # Verify
    check = pd.read_parquet(train_path)
    sample = check.iloc[0]["messages"]
    print(f"  Verify: type={type(sample)}, roles={[m['role'] for m in sample][:4]}...")
    
    return train_path, val_path


def prepare_grpo_parquet(grpo_data: List[Dict], output_dir: str,
                          val_ratio: float = 0.05, seed: int = 42):
    """Prepare GRPO parquet files.
    
    Normalizes two GRPO formats:
    1. Single-turn (has 'query'): prompt = [system, user_query]
    2. Multi-turn (has 'messages_prefix'): prompt = [system] + messages_prefix
    
    Both get converted to: prompt (list of dicts), reward_model, data_source, extra_info
    """
    import pandas as pd
    
    print(f"\n{'='*60}")
    print("Preparing GRPO Parquet")
    print(f"{'='*60}")
    
    records = []
    format_counts = Counter()
    
    for d in grpo_data:
        # Build prompt based on format
        if "query" in d:
            # Single-turn: system + user query
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": d["query"]},
            ]
            format_counts["single_turn"] += 1
            query_text = d["query"]
            
        elif "messages_prefix" in d:
            # Multi-turn: system + messages_prefix (which already has user + assistant + tool_result)
            prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + d["messages_prefix"]
            format_counts["multi_turn"] += 1
            # Extract the original user query from messages_prefix
            user_msgs = [m for m in d["messages_prefix"] if m["role"] == "user" 
                        and "<tool_result>" not in m.get("content", "")]
            query_text = user_msgs[0]["content"] if user_msgs else ""
        else:
            print(f"  WARNING: Skipping {d.get('id')} - no query or messages_prefix")
            continue
        
        # Determine safety label and expected triage
        safety_label = d.get("safety_label", "safe")
        expected_action = d.get("expected_action", "tool_call")
        category = d.get("category", "unknown")
        gold_answer = d.get("gold_answer", None)
        
        # Build ground_truth JSON string (this is what reward function reads)
        ground_truth = json.dumps({
            "gold_answer": gold_answer,
            "safety_label": safety_label,
            "category": category,
            "expected_triage": expected_action,
        })
        
        records.append({
            "prompt": prompt,  # list of dicts — NOT json.dumps
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "data_source": "triage_agent",
            "extra_info": {
                "episode_id": d.get("id", ""),
                "category": category,
                "query": query_text,
                "safety_label": safety_label,
            },
        })
    
    print(f"  Format distribution: {dict(format_counts)}")
    print(f"  Total records: {len(records)}")
    
    # Split
    random.seed(seed)
    random.shuffle(records)
    n_val = max(1, int(len(records) * val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]
    
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}")
    
    # Save
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    
    train_path = os.path.join(output_dir, "grpo_train.parquet")
    val_path = os.path.join(output_dir, "grpo_val.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"  Saved: {train_path} ({len(train_df)} rows)")
    print(f"  Saved: {val_path} ({len(val_df)} rows)")
    
    # Verify
    check = pd.read_parquet(train_path)
    print(f"  Verify columns: {list(check.columns)}")
    sample_prompt = check.iloc[0]["prompt"]
    print(f"  Verify prompt type: {type(sample_prompt)}")
    if hasattr(sample_prompt, '__len__'):
        print(f"  Verify prompt length: {len(sample_prompt)}")
        print(f"  Verify prompt[0] role: {sample_prompt[0].get('role', 'N/A') if isinstance(sample_prompt[0], dict) else type(sample_prompt[0])}")
    
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT + GRPO data for verl")
    parser.add_argument("--sft_dir", type=str, default="data/",
                        help="Directory containing sft_*.jsonl files")
    parser.add_argument("--grpo_dir", type=str, default="data/",
                        help="Directory containing grpo_*.jsonl files")
    parser.add_argument("--output_dir", type=str, default="data/",
                        help="Output directory for parquet files")
    parser.add_argument("--sft_val_ratio", type=float, default=0.1)
    parser.add_argument("--grpo_val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ============================================================
    # Load SFT data
    # ============================================================
    print("\n[Step 1] Loading SFT data...")
    sft_data = load_jsonl_files(args.sft_dir, "sft_0*.jsonl")
    
    if not sft_data:
        print("ERROR: No SFT data found!")
        sys.exit(1)
    
    print(f"\n  SFT total: {len(sft_data)}")
    sft_cats = Counter(d.get("category") for d in sft_data)
    for cat, count in sorted(sft_cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    
    # Validate
    print("\n[Step 2] Validating SFT data...")
    all_issues = []
    for i, d in enumerate(sft_data):
        all_issues.extend(validate_sft_example(d, i))
    print(f"  Issues found: {len(all_issues)}")
    for issue in all_issues[:10]:
        print(f"    {issue}")
    
    # ============================================================
    # Load GRPO data
    # ============================================================
    print("\n[Step 3] Loading GRPO data...")
    grpo_data = load_jsonl_files(args.grpo_dir, "grpo_*.jsonl")
    
    # Exclude GRPO_test_data.jsonl if it got mixed in
    grpo_data = [d for d in grpo_data if not d.get("id", "").startswith("test_")]
    
    if not grpo_data:
        print("ERROR: No GRPO data found!")
        sys.exit(1)
    
    print(f"\n  GRPO total: {len(grpo_data)}")
    grpo_cats = Counter(d.get("category") for d in grpo_data)
    for cat, count in sorted(grpo_cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    
    # Validate
    print("\n[Step 4] Validating GRPO data...")
    all_issues = []
    for i, d in enumerate(grpo_data):
        all_issues.extend(validate_grpo_example(d, i))
    print(f"  Issues found: {len(all_issues)}")
    for issue in all_issues[:10]:
        print(f"    {issue}")
    
    # ============================================================
    # Prepare parquets
    # ============================================================
    print("\n[Step 5] Preparing SFT parquet...")
    sft_train, sft_val = prepare_sft_parquet(sft_data, args.output_dir, 
                                              args.sft_val_ratio, args.seed)
    
    print("\n[Step 6] Preparing GRPO parquet...")
    grpo_train, grpo_val = prepare_grpo_parquet(grpo_data, args.output_dir,
                                                  args.grpo_val_ratio, args.seed)
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  SFT  train: {sft_train}")
    print(f"  SFT  val:   {sft_val}")
    print(f"  GRPO train: {grpo_train}")
    print(f"  GRPO val:   {grpo_val}")
    print(f"\nNext steps:")
    print(f"  1. SFT:  bash scripts/run_sft.sh  (1 epoch)")
    print(f"  2. Merge SFT checkpoint")
    print(f"  3. GRPO: bash scripts/run_grpo.sh  (MODEL_PATH=sft_merged)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
