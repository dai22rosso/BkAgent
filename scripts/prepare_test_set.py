"""
Prepare Test Set: merge 3 batch files into one validated JSONL

Usage:
    python scripts/prepare_test_set.py \
        --input_dir data/ \
        --output data/test_300.jsonl
"""

import argparse
import json
import os
import re
from collections import Counter


def fix_tool_result_json(content: str) -> str:
    """Fix non-JSON tool_result content."""
    match = re.search(r'<tool_result>(.*?)</tool_result>', content, re.DOTALL)
    if not match:
        return content
    inner = match.group(1).strip()
    try:
        json.loads(inner)
        return content
    except json.JSONDecodeError:
        fixed_json = json.dumps({"raw_result": inner})
        return content.replace(match.group(0), f"<tool_result>{fixed_json}</tool_result>")


def normalize_accept(accept_list):
    """Map any legacy action names in accept lists."""
    mapping = {"backtrack": "recover", "replan": "recover"}
    normalized = []
    for a in accept_list:
        normalized.append(mapping.get(a, a))
    return list(set(normalized))  # deduplicate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/")
    parser.add_argument("--output", type=str, default="data/test_300.jsonl")
    args = parser.parse_args()

    # Load all 3 batch files
    batch_files = sorted([
        os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir)
        if f.startswith("test_batch") and f.endswith(".jsonl")
    ])

    if not batch_files:
        print(f"ERROR: No test_batch*.jsonl files found in {args.input_dir}")
        return

    all_data = []
    for fp in batch_files:
        count = 0
        with open(fp) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line.strip())
                    all_data.append(d)
                    count += 1
        print(f"  Loaded {os.path.basename(fp)}: {count}")

    print(f"\nTotal loaded: {len(all_data)}")

    # Validate and normalize
    issues = 0
    for d in all_data:
        # Normalize accept lists (remove legacy backtrack/replan)
        if "accept" in d:
            d["accept"] = normalize_accept(d["accept"])

        # Fix tool_result JSON in multi-turn
        if d.get("type") == "multi_turn":
            for msg in d.get("history", []):
                if "<tool_result>" in msg.get("content", ""):
                    msg["content"] = fix_tool_result_json(msg["content"])

        # Validate required fields
        if d.get("type") == "single_turn" and "query" not in d:
            print(f"  WARNING: {d.get('id')} missing 'query'")
            issues += 1
        if d.get("type") == "multi_turn" and "history" not in d:
            print(f"  WARNING: {d.get('id')} missing 'history'")
            issues += 1

    # Check ID uniqueness
    ids = [d["id"] for d in all_data]
    if len(set(ids)) != len(ids):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        print(f"  WARNING: Duplicate IDs: {dupes}")
        issues += 1

    print(f"  Issues: {issues}")

    # Print distribution
    cats = Counter(d["category"] for d in all_data)
    print(f"\nCategory distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    actions = Counter(d["expected_action"] for d in all_data)
    print(f"\nExpected action distribution:")
    for act, count in sorted(actions.items(), key=lambda x: -x[1]):
        print(f"  {act}: {count}")

    # Write merged file
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for d in all_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nSaved: {args.output} ({len(all_data)} examples)")


if __name__ == "__main__":
    main()
