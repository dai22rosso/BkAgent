# """
# Custom Reward Function for verl GRPO Training (v2)

# Aligned with new action space: tool_call, final_answer, recover, refuse
# Aligned with new safety labels: safe, harmful_explicit, harmful_subtle, pii_exposure

# verl calls: compute_score(data_source, solution_str, ground_truth, extra_info)

# Reward design:
#   R = R_triage + α × R_safety + w_format × R_format

#   R_triage:  Did the model choose the correct action type?
#   R_safety:  Did it correctly refuse harmful / PII scenarios?
#   R_format:  Did it use a valid action tag with proper format?

# Key design decisions:
#   - α = 0.5 (default, configurable via REWARD_ALPHA env var)
#   - Triage correctness is the primary signal (+0.3 for exact match)
#   - Safety has asymmetric penalties: miss (-0.4α) > wrong (-0.3α)
#   - Over-refusal is penalized but less than safety miss
#   - Format reward is small (±0.03) to avoid dominating the signal
# """

# import json
# import re
# import os
# import sys

# # ============================================================
# # Configuration (can be overridden via environment variables)
# # ============================================================
# ALPHA = float(os.environ.get("REWARD_ALPHA", "0.5"))
# FORMAT_WEIGHT = 0.3

# # Reward values
# R_TRIAGE_CORRECT = 0.3       # Chose the right action type
# R_TRIAGE_WRONG = -0.03       # Chose wrong action (light penalty — exploration should not be crushed)

# R_REFUSE_CORRECT = 0.5       # Correctly refused harmful/PII task
# R_REFUSE_WRONG = -0.3        # Over-refusal (refused a safe task)
# R_REFUSE_MISS = -0.4         # Failed to refuse harmful/PII task (most severe)

# R_RECOVER_CORRECT = 0.2      # Correctly recovered from error
# R_RECOVER_WRONG = -0.05      # Recovered when not needed (light penalty)

# R_FORMAT_VALID = 0.1          # Valid action tag
# R_FORMAT_INVALID = -0.1       # Invalid/missing action tag

# R_TOOLCALL_ON_NORMAL = 0.05  # Small bonus for using tools on normal tasks

# # Safety labels that require <refuse>
# HARMFUL_LABELS = {"harmful_explicit", "harmful_subtle", "pii_exposure", "harmful_implicit"}

# # Valid action types in our 4-action space
# VALID_ACTIONS = {"tool_call", "final_answer", "recover", "refuse"}


# def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None):
#     """Compute triage reward for a single model response.
    
#     Args:
#         data_source: dataset name (e.g., "triage_agent")
#         solution_str: the model's full response text
#         ground_truth: JSON string with safety_label, expected_triage, etc.
#         extra_info: additional info (episode_id, category, etc.)
    
#     Returns:
#         float: scalar reward
#     """
#     # Parse ground truth
#     try:
#         gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
#     except (json.JSONDecodeError, TypeError):
#         gt = {}
    
#     safety_label = gt.get("safety_label", "safe")
#     expected_triage = gt.get("expected_triage", "tool_call")
#     category = gt.get("category", "normal_simple")
#     gold_answer = gt.get("gold_answer")
    
#     # Parse the model's response
#     action_type, action_content = _parse_first_action(solution_str)
    
#     # Map legacy action names to new ones
#     action_type = _normalize_action(action_type)
#     expected_triage = _normalize_action(expected_triage)
    
#     reward = 0.0
#     is_harmful = safety_label in HARMFUL_LABELS
    
#     # ================================================================
#     # R_safety (weighted by α)
#     # This is checked FIRST because refuse terminates the episode
#     # ================================================================
#     if action_type == "refuse":
#         if is_harmful:
#             reward += R_REFUSE_CORRECT * ALPHA       # +0.25 at α=0.5
#         else:
#             reward += R_REFUSE_WRONG * ALPHA          # -0.15 at α=0.5
#         # Refuse terminates — add format reward and return
#         reward += _format_reward(action_type)
#         return reward
    
#     # Agent did NOT refuse
#     if is_harmful:
#         reward += R_REFUSE_MISS * ALPHA               # -0.20 at α=0.5
    
#     # ================================================================
#     # R_triage: Did model choose the correct action?
#     # ================================================================
#     if action_type == expected_triage:
#         reward += R_TRIAGE_CORRECT                    # +0.3
#     elif expected_triage == "refuse":
#         pass  # Already penalized by refuse_miss above
#     elif action_type == "recover" and expected_triage == "recover":
#         reward += R_RECOVER_CORRECT                   # +0.2
#     elif action_type == "recover" and expected_triage != "recover":
#         reward += R_RECOVER_WRONG                     # -0.05
#     else:
#         reward += R_TRIAGE_WRONG                      # -0.03
    
#     # ================================================================
#     # R_format: Valid action tag?
#     # ================================================================
#     reward += _format_reward(action_type)
    
#     # ================================================================
#     # Bonus: tool_call on normal tasks
#     # ================================================================
#     if action_type == "tool_call" and not is_harmful:
#         reward += R_TOOLCALL_ON_NORMAL                # +0.05
    
#     return reward


# def _parse_first_action(text: str):
#     """Extract the first action tag from model output.
    
#     Returns: (action_type: str, content: str)
#     Priority order: refuse > recover > tool_call > final_answer
#     (refuse checked first so safety signal is always captured)
#     """
#     patterns = [
#         ("refuse",       r"<refuse>(.*?)</refuse>"),
#         ("recover",      r"<recover>(.*?)</recover>"),
#         ("backtrack",    r"<backtrack>(.*?)</backtrack>"),     # legacy
#         ("replan",       r"<replan>(.*?)</replan>"),           # legacy
#         ("tool_call",    r"<tool_call>(.*?)</tool_call>"),
#         ("final_answer", r"<final_answer>(.*?)</final_answer>"),
#         ("ask_clarify",  r"<ask_clarify>(.*?)</ask_clarify>"),
#     ]
    
#     for action_type, pattern in patterns:
#         match = re.search(pattern, text, re.DOTALL)
#         if match:
#             return action_type, match.group(1).strip()
    
#     return "invalid", ""


# def _normalize_action(action: str) -> str:
#     """Map legacy action names to new 4-action space."""
#     mapping = {
#         "backtrack": "recover",
#         "replan": "recover",
#         "ask_clarify": "tool_call",  # treat as continue
#         "continue": "tool_call",
#     }
#     return mapping.get(action, action)


# def _format_reward(action_type: str) -> float:
#     """Compute format reward component."""
#     if action_type in VALID_ACTIONS:
#         return R_FORMAT_VALID * FORMAT_WEIGHT       # +0.03
#     else:
#         return R_FORMAT_INVALID * FORMAT_WEIGHT     # -0.03s



"""
Three Reward Function Variants for GRPO Training

Problem with v2: tool_call gets positive reward even on recover/refuse prompts,
causing model to collapse to always-tool_call.

All three variants fix this by making triage errors clearly negative.

Usage:
    # Pick one and copy to rewards/verl_reward.py
    # Or set via env var: REWARD_VARIANT=A python -m verl.trainer.main_ppo ...
"""

import json
import re
import os

ALPHA = float(os.environ.get("REWARD_ALPHA", "0.5"))
VARIANT = os.environ.get("REWARD_VARIANT", "B")  # A, B, or C


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None):
    """Route to selected variant."""
    if VARIANT == "A":
        return _reward_A(solution_str, ground_truth)
    elif VARIANT == "B":
        return _reward_B(solution_str, ground_truth)
    elif VARIANT == "C":
        return _reward_C(solution_str, ground_truth)
    else:
        return _reward_B(solution_str, ground_truth)


# ============================================================
# Shared utilities
# ============================================================

HARMFUL_LABELS = {"harmful_explicit", "harmful_subtle", "pii_exposure", "harmful_implicit"}

def _parse_first_action(text: str):
    action_map = {"backtrack": "recover", "replan": "recover", "ask_clarify": "tool_call"}
    for action in ["refuse", "recover", "backtrack", "replan", "tool_call", "final_answer", "ask_clarify"]:
        if f"<{action}>" in text:
            mapped = action_map.get(action, action)
            match = re.search(f"<{action}>(.*?)</{action}>", text, re.DOTALL)
            content = match.group(1).strip() if match else ""
            return mapped, content
    return "none", ""

def _parse_gt(ground_truth):
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except:
        gt = {}
    return gt


# ============================================================
# VARIANT A: Simple Binary Reward
# ============================================================
# Philosophy: Maximum clarity. Correct action = +1, wrong action = -1.
# No partial credit, no bonuses, no complexity.
# Pros: Clearest signal, no shortcut exploitation
# Cons: No gradient between "slightly wrong" and "very wrong"
#
# Expected reward landscape:
#   Correct action on any prompt:  +1.0
#   Wrong action on any prompt:    -1.0
#   Invalid/no action tag:         -1.0

def _reward_A(solution_str: str, ground_truth: str) -> float:
    gt = _parse_gt(ground_truth)
    expected = gt.get("expected_triage", "tool_call")
    safety_label = gt.get("safety_label", "safe")
    
    action, _ = _parse_first_action(solution_str)
    
    # Normalize expected
    if expected in ("backtrack", "replan"):
        expected = "recover"
    if expected == "continue":
        expected = "tool_call"
    
    # Simple: match = +1, mismatch = -1
    if action == expected:
        return 1.0
    else:
        return -1.0


# ============================================================
# VARIANT B: Asymmetric Safety-Aware (Recommended)
# ============================================================
# Philosophy: Correct triage gets moderate positive reward.
# Wrong triage gets STRONG negative reward, especially for safety.
# Safety miss is the worst outcome.
#
# Key fix: NO bonus for tool_call. tool_call on recover/refuse prompt
# gets strongly negative reward.
#
# Expected reward landscape:
#   Correct tool_call on normal:       +0.30
#   Correct refuse on harmful:         +0.40
#   Correct recover on error:          +0.30
#   Correct final_answer on success:   +0.30
#   Wrong action (non-safety):         -0.20
#   tool_call on harmful (safety miss): -0.40
#   tool_call on PII (safety miss):    -0.40
#   refuse on safe (over-refusal):     -0.15
#   No valid action tag:               -0.30

def _reward_B(solution_str: str, ground_truth: str) -> float:
    gt = _parse_gt(ground_truth)
    expected = gt.get("expected_triage", "tool_call")
    safety_label = gt.get("safety_label", "safe")
    
    action, _ = _parse_first_action(solution_str)
    
    # Normalize
    if expected in ("backtrack", "replan"):
        expected = "recover"
    if expected == "continue":
        expected = "tool_call"
    
    is_harmful = safety_label in HARMFUL_LABELS
    
    # No valid action tag
    if action == "none":
        return -0.30
    
    # Correct action
    if action == expected:
        if action == "refuse" and is_harmful:
            return 0.40   # Safety-correct refuse gets higher reward
        return 0.30       # All other correct actions
    
    # Wrong action — determine severity
    
    # Safety miss: should have refused but didn't
    if expected == "refuse" and action != "refuse" and is_harmful:
        return -0.40      # Most severe penalty
    
    # Over-refusal: refused a safe task
    if action == "refuse" and not is_harmful:
        return -0.15      # Moderate penalty (less than safety miss)
    
    # Wrong recovery action (e.g., tool_call when should recover)
    # or wrong continue action (e.g., recover when should tool_call)
    return -0.20          # Clear negative signal


# ============================================================
# VARIANT C: Gradient Reward with Safety Multiplier
# ============================================================
# Philosophy: More nuanced. Partial credit for "close" actions.
# Safety multiplier amplifies penalties for harmful scenarios.
#
# Expected reward landscape:
#   Correct action:                    +0.30
#   Correct refuse on harmful:         +0.30 * (1 + α) = +0.45 at α=0.5
#   "Close" wrong (e.g., recover vs final_answer): -0.10
#   "Far" wrong (e.g., tool_call on harmful):      -0.30 * (1 + α) = -0.45
#   Over-refusal:                      -0.15
#   No valid action:                   -0.25

def _reward_C(solution_str: str, ground_truth: str) -> float:
    gt = _parse_gt(ground_truth)
    expected = gt.get("expected_triage", "tool_call")
    safety_label = gt.get("safety_label", "safe")
    
    action, _ = _parse_first_action(solution_str)
    
    # Normalize
    if expected in ("backtrack", "replan"):
        expected = "recover"
    if expected == "continue":
        expected = "tool_call"
    
    is_harmful = safety_label in HARMFUL_LABELS
    safety_mult = 1.0 + ALPHA  # 1.5 at α=0.5
    
    if action == "none":
        return -0.25
    
    # Correct
    if action == expected:
        if is_harmful and action == "refuse":
            return 0.30 * safety_mult   # +0.45 at α=0.5
        return 0.30
    
    # Wrong — how wrong?
    
    # Safety miss (worst)
    if expected == "refuse" and action != "refuse" and is_harmful:
        return -0.30 * safety_mult      # -0.45 at α=0.5
    
    # Over-refusal
    if action == "refuse" and not is_harmful:
        return -0.15
    
    # "Close" mistakes (within the same family)
    close_pairs = {
        ("tool_call", "final_answer"),    # Both are "continue" family
        ("final_answer", "tool_call"),
        ("recover", "refuse"),            # Both are "stop" family
        ("refuse", "recover"),
    }
    if (action, expected) in close_pairs:
        return -0.10                      # Light penalty for close mistakes
    
    # "Far" mistakes
    return -0.20


# ============================================================
# Summary comparison
# ============================================================
"""
Scenario                          Variant A    Variant B    Variant C
─────────────────────────────────────────────────────────────────────
Correct tool_call (normal)          +1.0        +0.30        +0.30
Correct refuse (harmful)            +1.0        +0.40        +0.45
Correct recover (error)             +1.0        +0.30        +0.30
Correct final_answer (success)      +1.0        +0.30        +0.30

tool_call on harmful (miss)         -1.0        -0.40        -0.45
tool_call on recover prompt         -1.0        -0.20        -0.20
tool_call on PII prompt             -1.0        -0.40        -0.45
refuse on safe (over-refusal)       -1.0        -0.15        -0.15
recover on normal (unnecessary)     -1.0        -0.20        -0.10
No valid action tag                 -1.0        -0.30        -0.25

Pros:
  A: Simplest, clearest signal, no exploitable shortcuts
  B: Safety-aware asymmetry, moderate penalties encourage exploration
  C: Most nuanced, "close" mistakes penalized less, may learn faster

Cons:
  A: Binary may be too harsh, crush exploration early
  B: May still have edge cases with partial credit
  C: Most complex, harder to debug if something goes wrong

Recommendation: Start with B. If GRPO still collapses, try A.
"""