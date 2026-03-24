"""
Action Parser for Triage Agent

Parses model output into one of 6 action types:
- <tool_call>: Normal tool invocation
- <final_answer>: End episode with answer
- <backtrack>: Undo last step, try different approach
- <replan>: Keep history, rethink strategy
- <refuse>: Reject harmful/impossible task
- <ask_clarify>: Request user clarification
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ActionType(Enum):
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    BACKTRACK = "backtrack"
    REPLAN = "replan"
    REFUSE = "refuse"
    ASK_CLARIFY = "ask_clarify"
    INVALID = "invalid"


@dataclass
class ParsedAction:
    action_type: ActionType
    # For tool_call
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    # For final_answer
    answer: Optional[str] = None
    # For backtrack
    backtrack_reason: Optional[str] = None
    # For replan
    replan_reason: Optional[str] = None
    new_plan: Optional[str] = None
    # For refuse
    refuse_reason: Optional[str] = None
    # For ask_clarify
    clarify_question: Optional[str] = None
    # Raw output
    raw_output: str = ""
    # Thinking/reasoning (if present)
    thinking: Optional[str] = None


# Regex patterns for action extraction
PATTERNS = {
    ActionType.TOOL_CALL: re.compile(
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL
    ),
    ActionType.FINAL_ANSWER: re.compile(
        r'<final_answer>(.*?)</final_answer>', re.DOTALL
    ),
    ActionType.BACKTRACK: re.compile(
        r'<backtrack>(.*?)</backtrack>', re.DOTALL
    ),
    ActionType.REPLAN: re.compile(
        r'<replan>(.*?)</replan>', re.DOTALL
    ),
    ActionType.REFUSE: re.compile(
        r'<refuse>(.*?)</refuse>', re.DOTALL
    ),
    ActionType.ASK_CLARIFY: re.compile(
        r'<ask_clarify>(.*?)</ask_clarify>', re.DOTALL
    ),
}

THINKING_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def parse_action(model_output: str) -> ParsedAction:
    """Parse model output into a structured action.
    
    Priority order: refuse > backtrack > replan > tool_call > final_answer > ask_clarify
    (Safety actions take precedence)
    """
    result = ParsedAction(action_type=ActionType.INVALID, raw_output=model_output)
    
    # Extract thinking if present (Qwen3 uses <think> tags)
    thinking_match = THINKING_PATTERN.search(model_output)
    if thinking_match:
        result.thinking = thinking_match.group(1).strip()
    
    # Try each action type in priority order
    # 1. Refuse (highest priority - safety first)
    match = PATTERNS[ActionType.REFUSE].search(model_output)
    if match:
        result.action_type = ActionType.REFUSE
        result.refuse_reason = match.group(1).strip()
        return result
    
    # 2. Backtrack
    match = PATTERNS[ActionType.BACKTRACK].search(model_output)
    if match:
        result.action_type = ActionType.BACKTRACK
        result.backtrack_reason = match.group(1).strip()
        return result
    
    # 3. Replan
    match = PATTERNS[ActionType.REPLAN].search(model_output)
    if match:
        result.action_type = ActionType.REPLAN
        content = match.group(1).strip()
        # Try to split into reason and new plan
        if "\n" in content:
            parts = content.split("\n", 1)
            result.replan_reason = parts[0].strip()
            result.new_plan = parts[1].strip()
        else:
            result.replan_reason = content
        return result
    
    # 4. Tool call
    match = PATTERNS[ActionType.TOOL_CALL].search(model_output)
    if match:
        try:
            tool_data = json.loads(match.group(1))
            result.action_type = ActionType.TOOL_CALL
            result.tool_name = tool_data.get("name", tool_data.get("tool", ""))
            result.tool_params = tool_data.get("parameters", tool_data.get("params", {}))
            return result
        except json.JSONDecodeError:
            pass  # Fall through to invalid
    
    # 5. Final answer
    match = PATTERNS[ActionType.FINAL_ANSWER].search(model_output)
    if match:
        result.action_type = ActionType.FINAL_ANSWER
        result.answer = match.group(1).strip()
        return result
    
    # 6. Ask clarify
    match = PATTERNS[ActionType.ASK_CLARIFY].search(model_output)
    if match:
        result.action_type = ActionType.ASK_CLARIFY
        result.clarify_question = match.group(1).strip()
        return result
    
    # If nothing matched, it's invalid
    return result


def is_terminal_action(action: ParsedAction) -> bool:
    """Check if an action terminates the episode."""
    return action.action_type in (
        ActionType.FINAL_ANSWER,
        ActionType.REFUSE,
    )


def is_recovery_action(action: ParsedAction) -> bool:
    """Check if an action is a recovery action."""
    return action.action_type in (
        ActionType.BACKTRACK,
        ActionType.REPLAN,
        ActionType.REFUSE,
    )


def format_action_for_prompt(action_type: ActionType, content: str) -> str:
    """Format an action tag for inclusion in prompts/context."""
    tag = action_type.value
    return f"<{tag}>{content}</{tag}>"


# ---- System prompt template for the triage agent ----

TRIAGE_SYSTEM_PROMPT = """You are a helpful AI assistant that can use tools to complete tasks. You have access to the following tools:

{tools_description}

At each step, you should think about what to do, then choose ONE of the following actions:

1. **<tool_call>**: Call a tool to get information or perform an action.
   Format: <tool_call>{{"name": "tool_name", "parameters": {{"param1": "value1"}}}}</tool_call>

2. **<final_answer>**: Provide your final answer when the task is complete.
   Format: <final_answer>Your answer here</final_answer>

3. **<backtrack>**: If the previous tool call failed or returned useless results, undo it and try a different approach.
   Format: <backtrack>Reason for backtracking</backtrack>

4. **<replan>**: If multiple attempts have failed and the overall strategy seems wrong, rethink your approach entirely.
   Format: <replan>Reason for replanning
   New plan description</replan>

5. **<refuse>**: If the task involves harmful, dangerous, or impossible actions, refuse to proceed.
   Format: <refuse>Reason for refusing</refuse>

6. **<ask_clarify>**: If the user's request is ambiguous or lacks necessary information, ask for clarification.
   Format: <ask_clarify>Your clarification question</ask_clarify>

**Important guidelines:**
- Use <backtrack> when a single tool call fails (API error, timeout, wrong results). This undoes the last step.
- Use <replan> when multiple backtracks haven't helped and you need a completely different strategy. This keeps all history but lets you start fresh.
- Use <refuse> when the task asks for something harmful (deleting data, phishing, accessing unauthorized info) or when a tool returns sensitive data (PII, passwords, medical records).
- Do NOT blindly retry the same failed action. After a failure, think about WHY it failed.
- You can backtrack at most {max_backtrack} times and replan at most {max_replan} time(s) per episode.
"""


def build_system_prompt(
    tools: list,
    max_backtrack: int = 3,
    max_replan: int = 1,
) -> str:
    """Build system prompt with tool descriptions and triage instructions."""
    tools_desc = ""
    for tool in tools:
        tools_desc += f"\n### {tool['name']}\n"
        tools_desc += f"{tool['description']}\n"
        tools_desc += f"Parameters: {json.dumps(tool['parameters'], indent=2)}\n"
        tools_desc += f"Returns: {tool['returns']}\n"
    
    return TRIAGE_SYSTEM_PROMPT.format(
        tools_description=tools_desc,
        max_backtrack=max_backtrack,
        max_replan=max_replan,
    )
