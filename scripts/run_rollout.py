"""
Rollout Script — Run episodes with a model to collect trajectories.

This is the core inference loop:
1. Load model (Qwen3-4B-Instruct) via vLLM
2. Load episodes from JSONL
3. For each episode: reset env → model generates → env steps → repeat until done
4. Collect trajectories + rewards

Usage:
    python scripts/run_rollout.py \
        --model ~/models/Qwen3-4B-Instruct \
        --data data/train_demo.jsonl \
        --max_episodes 10 \
        --output output/rollouts.jsonl
"""

import argparse
import json
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.triage_tool_env import TriageToolEnv, EnvConfig
from utils.action_parser import parse_action, ActionType

logger = logging.getLogger(__name__)


def load_episodes(path: str) -> list:
    """Load episodes from JSONL file."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def create_model(model_path: str, gpu_memory_utilization: float = 0.8):
    """Create vLLM model for inference.
    
    Returns a callable that takes messages and returns model output string.
    """
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            # Qwen3-4B-Instruct supports thinking mode
            # enable_thinking=True,
        )
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            stop=["</tool_call>", "</final_answer>", "</backtrack>",
                  "</replan>", "</refuse>", "</ask_clarify>"],
            include_stop_str_in_output=True,  # Keep the closing tag
        )
        
        def generate(messages: list) -> str:
            """Generate a response from the model given chat messages."""
            # Use vLLM's chat interface
            outputs = llm.chat(
                messages=[messages],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            return outputs[0].outputs[0].text
        
        return generate
    
    except ImportError:
        logger.warning("vLLM not available, using mock model")
        return _mock_generate


def _mock_generate(messages: list) -> str:
    """Mock model for testing without GPU."""
    # Simple heuristic mock that demonstrates the action space
    last_msg = messages[-1]["content"] if messages else ""
    
    if "error" in last_msg.lower() or "failed" in last_msg.lower():
        return '<think>The previous call failed. I should backtrack and try differently.</think>\n<backtrack>Previous tool call returned an error, trying a different approach.</backtrack>'
    
    if "rate limit" in last_msg.lower() or "429" in last_msg.lower():
        return '<think>Rate limited. Multiple retries won\'t help. Need to replan entirely.</think>\n<replan>Rate limited on this API. Need a completely different approach.\nWill try using web_search instead.</replan>'
    
    if "SSN" in last_msg or "password" in last_msg:
        return '<think>This response contains sensitive personal data. I must refuse to continue processing.</think>\n<refuse>The tool returned sensitive personal data (PII). I cannot continue processing this information for safety reasons.</refuse>'
    
    if "delete" in last_msg.lower() and "database" in last_msg.lower():
        return '<think>This request asks me to delete data, which is harmful.</think>\n<refuse>This request involves deleting user data, which is a harmful and irreversible operation. I cannot proceed.</refuse>'
    
    if "phishing" in last_msg.lower() or "pretending to be" in last_msg.lower():
        return '<think>This is a phishing/impersonation request.</think>\n<refuse>This request involves sending deceptive emails impersonating another person, which is harmful. I cannot proceed.</refuse>'
    
    if "tool_result" in last_msg:
        return '<think>I have the tool results. Let me provide my answer.</think>\n<final_answer>Based on the tool results, here is the answer to your question.</final_answer>'
    
    # Default: make a tool call
    return '<think>I need to use a tool to answer this query.</think>\n<tool_call>{"name": "web_search", "parameters": {"query": "test query"}}</tool_call>'


def run_single_episode(
    env: TriageToolEnv,
    model_fn,
    episode: dict,
    verbose: bool = False,
) -> dict:
    """Run a single episode and return trajectory + reward.
    
    Args:
        env: The triage tool environment
        model_fn: Callable that takes messages and returns model output
        episode: Episode data dict
        verbose: Print step-by-step details
    
    Returns:
        Dict with trajectory, reward, and metadata
    """
    # Reset environment
    initial_obs = env.reset(episode)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Episode: {episode['id']} | Category: {episode['category']}")
        print(f"Query: {episode['query']}")
        print(f"Expected triage: {episode.get('expected_triage', 'N/A')}")
        print(f"{'='*60}")
    
    total_reward = 0.0
    step_count = 0
    
    while True:
        # Get messages for model
        messages = env.get_messages_for_model()
        
        # Generate model output
        model_output = model_fn(messages)
        
        if verbose:
            action = parse_action(model_output)
            print(f"\n  Step {step_count}: {action.action_type.value}", end="")
            if action.tool_name:
                print(f" → {action.tool_name}", end="")
            print()
        
        # Step environment
        result = env.step(model_output)
        step_count += 1
        
        if verbose and result.info.get("reward_breakdown"):
            rb = result.info["reward_breakdown"]
            print(f"  Reward: {result.reward:.3f} "
                  f"(task={rb['r_task']:.2f}, recovery={rb['r_recovery']:.2f}, "
                  f"safety={rb['r_safety']:.2f})")
        
        if result.done:
            total_reward = result.reward
            break
    
    # Collect results
    trajectory = env.get_trajectory()
    
    result_data = {
        "episode_id": episode["id"],
        "category": episode["category"],
        "safety_label": episode.get("safety_label", "safe"),
        "expected_triage": episode.get("expected_triage"),
        "total_reward": total_reward,
        "steps": step_count,
        "trajectory": trajectory,
        "info": result.info,
    }
    
    if verbose:
        print(f"\n  Final reward: {total_reward:.3f}")
        print(f"  Steps: {step_count}")
        if result.info.get("trajectory_stats"):
            stats = result.info["trajectory_stats"]
            print(f"  Backtracks: {stats['backtrack_count']}, Replans: {stats['replan_count']}")
    
    return result_data


def main():
    parser = argparse.ArgumentParser(description="Run rollout episodes")
    parser.add_argument("--model", type=str, default="~/models/Qwen3-4B-Instruct",
                        help="Path to model")
    parser.add_argument("--data", type=str, default="data/train_demo.jsonl",
                        help="Path to episode data")
    parser.add_argument("--output", type=str, default="output/rollouts.jsonl",
                        help="Output path for rollout results")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum number of episodes to run")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock model (no GPU needed)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print step-by-step details")
    parser.add_argument("--tools_registry", type=str, default="data/tools_registry.json",
                        help="Path to tools registry")
    parser.add_argument("--max_turns", type=int, default=12)
    parser.add_argument("--max_backtrack", type=int, default=3)
    parser.add_argument("--max_replan", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Safety reward weight")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # Load episodes
    episodes = load_episodes(args.data)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    print(f"Loaded {len(episodes)} episodes from {args.data}")
    
    # Create model
    if args.mock:
        model_fn = _mock_generate
        print("Using mock model (no GPU)")
    else:
        model_path = os.path.expanduser(args.model)
        print(f"Loading model from {model_path}...")
        model_fn = create_model(model_path)
    
    # Create environment
    env_config = EnvConfig(
        max_turns=args.max_turns,
        max_backtrack=args.max_backtrack,
        max_replan=args.max_replan,
        reward_alpha=args.alpha,
        tools_registry_path=args.tools_registry,
    )
    env = TriageToolEnv(env_config)
    
    # Run episodes
    results = []
    for i, episode in enumerate(episodes):
        print(f"\nRunning episode {i+1}/{len(episodes)}: {episode['id']}")
        result = run_single_episode(env, model_fn, episode, verbose=args.verbose)
        results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in results:
            # Simplify for JSON serialization
            simplified = {
                "episode_id": r["episode_id"],
                "category": r["category"],
                "safety_label": r["safety_label"],
                "expected_triage": r["expected_triage"],
                "total_reward": r["total_reward"],
                "steps": r["steps"],
            }
            if r["info"].get("reward_breakdown"):
                simplified["reward_breakdown"] = r["info"]["reward_breakdown"]
            if r["info"].get("trajectory_stats"):
                simplified["trajectory_stats"] = r["info"]["trajectory_stats"]
            f.write(json.dumps(simplified) + "\n")
    
    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    print(f"{'='*60}")
    
    # Summary
    rewards = [r["total_reward"] for r in results]
    print(f"Episodes: {len(results)}")
    print(f"Avg reward: {sum(rewards)/len(rewards):.3f}")
    print(f"Max reward: {max(rewards):.3f}")
    print(f"Min reward: {min(rewards):.3f}")
    
    # Per-category summary
    from collections import defaultdict
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["total_reward"])
    
    print(f"\nPer-category avg reward:")
    for cat, rews in sorted(by_category.items()):
        print(f"  {cat}: {sum(rews)/len(rews):.3f} (n={len(rews)})")


if __name__ == "__main__":
    main()
