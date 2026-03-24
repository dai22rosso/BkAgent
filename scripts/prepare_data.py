"""
Data Preprocessing: JSONL → Parquet for verl

verl requires parquet files with specific fields:
- prompt: The chat-formatted prompt (list of message dicts, or string)
- reward_model.ground_truth: Ground truth for reward computation
- data_source: Dataset identifier

This script converts our train_demo.jsonl / train_100.jsonl into verl-compatible parquet.

Usage:
    python scripts/prepare_data.py \
        --input data/train_demo.jsonl \
        --output data/train_demo.parquet \
        --tools_registry data/tools_registry.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.action_parser import build_system_prompt


def load_tools(path: str) -> list:
    try:
        with open(path) as f:
            return json.load(f).get("tools", [])
    except FileNotFoundError:
        return []


def jsonl_to_parquet(
    input_path: str,
    output_path: str,
    tools_registry_path: str = "data/tools_registry.json",
    max_backtrack: int = 3,
    max_replan: int = 1,
):
    """Convert JSONL episodes to verl-compatible parquet."""
    import pandas as pd

    all_tools = load_tools(tools_registry_path)

    records = []
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line.strip())

            # Filter tools for this episode
            ep_tools = [t for t in all_tools if t["name"] in ep.get("tools_available", [])]
            if not ep_tools:
                ep_tools = all_tools

            system_prompt = build_system_prompt(
                tools=ep_tools,
                max_backtrack=max_backtrack,
                max_replan=max_replan,
            )

            # verl expects 'prompt' as a JSON-serialized list of messages
            prompt = json.dumps([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ep["query"]},
            ])

            # ground_truth is a JSON string containing everything the reward
            # function needs to compute the score
            ground_truth = json.dumps({
                "gold_answer": ep.get("gold_answer"),
                "safety_label": ep.get("safety_label", "safe"),
                "category": ep.get("category", "normal"),
                "expected_triage": ep.get("expected_triage"),
                "failure_injection": ep.get("failure_injection", {}),
                "tools_available": ep.get("tools_available", []),
            })

            records.append({
                # prompt 保持原样，verl 支持字符串格式的 prompt
                "prompt": prompt, 
                
                # 去掉 json.dumps
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                
                "data_source": "triage_agent",
                
                # 去掉 json.dumps，传真实的 dict 进去
                "extra_info": {
                    "episode_id": ep["id"],
                    "category": ep.get("category"),
                    "query": ep["query"],
                },
            })

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Converted {len(records)} episodes: {input_path} → {output_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to verl parquet")
    parser.add_argument("--input", type=str, default="data/train_demo.jsonl")
    parser.add_argument("--output", type=str, default="data/train_demo.parquet")
    parser.add_argument("--tools_registry", type=str, default="data/tools_registry.json")
    args = parser.parse_args()

    jsonl_to_parquet(args.input, args.output, args.tools_registry)


if __name__ == "__main__":
    main()
