"""
Triage Decision Matrix Analysis

This generates the triage decision matrix — a key figure in the paper that answers:
"Does the agent learn to use the RIGHT recovery action for each failure type?"

Rows: Failure types (API Error, Timeout, Misleading, Path Block, Harmful, None)
Cols: Agent actions (continue, backtrack, replan, refuse)

Also produces:
- Recovery timing analysis (early vs late recovery)
- Per-failure-type recovery success rate
- Confusion matrix for triage decisions
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_triage_matrix(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the triage decision matrix from rollout results.
    
    Returns:
        Dict with matrix, per-failure stats, and timing analysis.
    """
    # Matrix: failure_type → action_taken → count
    matrix = defaultdict(lambda: defaultdict(int))
    
    # Per-failure success rate
    failure_recovery = defaultdict(lambda: {"total": 0, "recovered": 0})
    
    # Timing: where in the trajectory did recovery happen?
    timing = {"early": 0, "mid": 0, "late": 0, "total_recovery": 0}
    
    for r in rollouts:
        # Determine failure type from episode data
        # We infer from expected_triage and category
        expected = r.get("expected_triage", "continue")
        category = r.get("category", "normal")
        safety_label = r.get("safety_label", "safe")
        
        # Map to failure type
        if safety_label in ("harmful_explicit", "harmful_implicit"):
            failure_type = safety_label
        elif expected == "replan":
            failure_type = "path_block"
        elif expected == "backtrack":
            failure_type = "api_failure"  # Could be error/timeout/misleading
        elif expected == "refuse" and category == "impossible":
            failure_type = "impossible"
        else:
            failure_type = "none"
        
        # Determine actual action taken
        stats = r.get("trajectory_stats", {})
        rb = r.get("reward_breakdown", {})
        
        if rb.get("refuse_used", False):
            actual_action = "refuse"
        elif stats.get("replan_count", 0) > 0:
            actual_action = "replan"
        elif stats.get("backtrack_count", 0) > 0:
            actual_action = "backtrack"
        else:
            actual_action = "continue"
        
        matrix[failure_type][actual_action] += 1
        
        # Recovery stats
        if actual_action in ("backtrack", "replan"):
            failure_recovery[failure_type]["total"] += 1
            if rb.get("task_success", False):
                failure_recovery[failure_type]["recovered"] += 1
        
        # Timing analysis
        recovery_positions = stats.get("recovery_positions", [])
        if recovery_positions:
            timing["total_recovery"] += 1
            avg_pos = sum(recovery_positions) / len(recovery_positions)
            if avg_pos < 0.33:
                timing["early"] += 1
            elif avg_pos < 0.66:
                timing["mid"] += 1
            else:
                timing["late"] += 1
    
    return {
        "matrix": {k: dict(v) for k, v in matrix.items()},
        "failure_recovery": {k: dict(v) for k, v in failure_recovery.items()},
        "timing": timing,
    }


def print_triage_matrix(analysis: Dict[str, Any]):
    """Pretty-print the triage decision matrix."""
    matrix = analysis["matrix"]
    
    print("\n" + "=" * 70)
    print("TRIAGE DECISION MATRIX")
    print("(Rows: failure type, Cols: agent action)")
    print("=" * 70)
    
    all_actions = ["continue", "backtrack", "replan", "refuse"]
    all_failures = sorted(matrix.keys())
    
    # Header
    header = f"{'Failure Type':<20}" + "".join(f"{a:<12}" for a in all_actions) + "Total"
    print(header)
    print("-" * len(header))
    
    for ft in all_failures:
        row_data = matrix.get(ft, {})
        total = sum(row_data.values())
        row = f"{ft:<20}"
        for act in all_actions:
            count = row_data.get(act, 0)
            pct = f"{count}({100*count/total:.0f}%)" if total > 0 else "0"
            row += f"{pct:<12}"
        row += str(total)
        print(row)
    
    # Recovery success per failure type
    print(f"\n--- Recovery Success Rate by Failure Type ---")
    fr = analysis["failure_recovery"]
    for ft in sorted(fr.keys()):
        data = fr[ft]
        total = data["total"]
        recovered = data["recovered"]
        rate = recovered / total if total > 0 else 0
        print(f"  {ft}: {recovered}/{total} = {rate:.1%}")
    
    # Timing
    print(f"\n--- Recovery Timing ---")
    t = analysis["timing"]
    total_r = t["total_recovery"]
    if total_r > 0:
        print(f"  Early (0-33% of trajectory): {t['early']}/{total_r} = {t['early']/total_r:.1%}")
        print(f"  Mid (33-66%):                {t['mid']}/{total_r} = {t['mid']/total_r:.1%}")
        print(f"  Late (66-100%):              {t['late']}/{total_r} = {t['late']/total_r:.1%}")
    else:
        print("  No recovery actions observed.")
    
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Triage decision matrix analysis")
    parser.add_argument("--rollouts", type=str, required=True, help="Path to rollouts JSONL")
    parser.add_argument("--output", type=str, default="output/triage_analysis.json")
    args = parser.parse_args()
    
    # Load rollouts
    rollouts = []
    with open(args.rollouts) as f:
        for line in f:
            if line.strip():
                rollouts.append(json.loads(line.strip()))
    
    # Build analysis
    analysis = build_triage_matrix(rollouts)
    
    # Print
    print_triage_matrix(analysis)
    
    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
