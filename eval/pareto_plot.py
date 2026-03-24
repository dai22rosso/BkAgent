"""
Safety-Task Pareto Curve Generator

This is the most important figure in the full version paper.
It shows the tradeoff between safety refusal rate and task success rate
across different α values (safety reward weight).

Expected shape:
- α=0: High task success, low safety refusal (RL erodes safety)
- α=0.3: Good task, moderate safety
- α=0.5: Balanced (our default)
- α=0.7: Lower task, high safety
- α=1.0: Lowest task, highest safety (over-refuses)

The Pareto frontier shows optimal tradeoff configurations.

Usage:
    python eval/pareto_plot.py \
        --results output/alpha_0.json output/alpha_03.json ... \
        --alphas 0.0 0.3 0.5 0.7 1.0 \
        --output output/pareto_curve.png
"""

import json
import os
import sys
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_pareto_points(
    alpha_results: List[Tuple[float, Dict]],
) -> List[Dict]:
    """Extract (task_success_rate, safety_refusal_rate) for each α.
    
    Args:
        alpha_results: List of (alpha_value, eval_metrics_dict) pairs
    
    Returns:
        List of points for plotting
    """
    points = []
    for alpha, metrics in alpha_results:
        point = {
            "alpha": alpha,
            "task_success_rate": metrics.get("task_success_rate", 0),
            "safety_refusal_rate": metrics.get("safety_refusal_rate", 0),
            "over_refusal_rate": metrics.get("over_refusal_rate", 0),
            "avg_reward": metrics.get("reward_stats", {}).get("mean", 0),
        }
        points.append(point)
    return points


def find_pareto_frontier(points: List[Dict]) -> List[Dict]:
    """Find Pareto-optimal points (maximize both task success and safety).
    
    A point is Pareto-optimal if no other point dominates it
    (i.e., no point is better in BOTH dimensions).
    """
    frontier = []
    for p in points:
        dominated = False
        for q in points:
            if (q["task_success_rate"] >= p["task_success_rate"]
                    and q["safety_refusal_rate"] >= p["safety_refusal_rate"]
                    and (q["task_success_rate"] > p["task_success_rate"]
                         or q["safety_refusal_rate"] > p["safety_refusal_rate"])):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    
    # Sort by task success rate for plotting
    frontier.sort(key=lambda x: x["task_success_rate"])
    return frontier


def generate_plot_text(points: List[Dict], frontier: List[Dict]) -> str:
    """Generate a text-based representation of the Pareto curve.
    
    For environments without matplotlib, this provides a readable summary.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SAFETY-TASK PARETO CURVE")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'α':<8}{'Task SR':<12}{'Safety RR':<12}{'Over-Refuse':<12}{'Pareto?'}")
    lines.append("-" * 56)
    
    frontier_alphas = {p["alpha"] for p in frontier}
    
    for p in sorted(points, key=lambda x: x["alpha"]):
        is_pareto = "★" if p["alpha"] in frontier_alphas else ""
        lines.append(
            f"{p['alpha']:<8.1f}"
            f"{p['task_success_rate']:<12.1%}"
            f"{p['safety_refusal_rate'] or 0:<12.1%}"
            f"{p['over_refusal_rate'] or 0:<12.1%}"
            f"{is_pareto}"
        )
    
    lines.append("")
    lines.append("★ = Pareto-optimal (no point dominates in both dimensions)")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Low α: Agent prioritizes task completion, may miss harmful tasks")
    lines.append("  - High α: Agent prioritizes safety, may over-refuse safe tasks")
    lines.append("  - Pareto points represent optimal tradeoff configurations")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def generate_matplotlib_plot(
    points: List[Dict],
    frontier: List[Dict],
    output_path: str,
):
    """Generate the actual Pareto curve plot with matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # All points
    task_rates = [p["task_success_rate"] for p in points]
    safety_rates = [p["safety_refusal_rate"] or 0 for p in points]
    alphas = [p["alpha"] for p in points]
    
    scatter = ax.scatter(task_rates, safety_rates, c=alphas, cmap='RdYlGn',
                         s=100, zorder=3, edgecolors='black', linewidths=1)
    
    # Label each point with α value
    for p in points:
        ax.annotate(
            f'α={p["alpha"]}',
            (p["task_success_rate"], p["safety_refusal_rate"] or 0),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )
    
    # Pareto frontier
    if len(frontier) > 1:
        front_task = [p["task_success_rate"] for p in frontier]
        front_safety = [p["safety_refusal_rate"] or 0 for p in frontier]
        ax.plot(front_task, front_safety, 'r--', linewidth=2, label='Pareto Frontier',
                zorder=2, alpha=0.7)
    
    ax.set_xlabel('Task Success Rate', fontsize=12)
    ax.set_ylabel('Safety Refusal Rate', fontsize=12)
    ax.set_title('Safety-Task Pareto Curve\n(varying α: safety reward weight)', fontsize=14)
    
    plt.colorbar(scatter, ax=ax, label='α (safety weight)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Safety-Task Pareto curve")
    parser.add_argument("--results", nargs="+", required=True,
                        help="Paths to eval result JSON files (one per α)")
    parser.add_argument("--alphas", nargs="+", type=float, required=True,
                        help="α values corresponding to each result file")
    parser.add_argument("--output", type=str, default="output/pareto_curve.png")
    args = parser.parse_args()
    
    assert len(args.results) == len(args.alphas), "Must provide same number of results and alphas"
    
    # Load results
    alpha_results = []
    for alpha, path in zip(args.alphas, args.results):
        with open(path) as f:
            metrics = json.load(f)
        alpha_results.append((alpha, metrics))
    
    # Compute points
    points = compute_pareto_points(alpha_results)
    frontier = find_pareto_frontier(points)
    
    # Text output
    text_report = generate_plot_text(points, frontier)
    print(text_report)
    
    # Plot
    generate_matplotlib_plot(points, frontier, args.output)
    
    # Save data
    data_path = args.output.replace(".png", "_data.json")
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    with open(data_path, "w") as f:
        json.dump({
            "points": points,
            "frontier": frontier,
        }, f, indent=2)
    print(f"Data saved to {data_path}")


if __name__ == "__main__":
    main()
