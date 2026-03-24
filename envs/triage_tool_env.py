"""
Triage Tool Environment for verl

This is the core environment that integrates with verl's RL training loop.
It manages multi-turn tool-use episodes with 4-action triage:
  continue (tool_call / final_answer) | backtrack | replan | refuse

Key responsibilities:
1. Parse agent actions (6 types)
2. Execute tool calls with failure injection
3. Handle state rollback (backtrack) and replan
4. Compute rewards via TriageRewardFunction
5. Manage episode lifecycle (termination, max turns, etc.)

Integration with verl:
- This env is used inside verl's rollout worker
- Each episode = one prompt → multi-turn trajectory → reward
- GRPO collects K rollouts per prompt and uses reward for policy gradient
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from ..utils.action_parser import (
        ActionType, ParsedAction, parse_action,
        is_terminal_action, build_system_prompt,
    )
    from ..utils.trajectory_utils import EpisodeState, compute_trajectory_stats
    from .failure_injection import FailureInjector, create_injector_from_episode, FailureType
    from .user_simulator import simulate_user_response
    from ..rewards.triage_reward import TriageRewardFunction, RewardBreakdown, get_reward_for_verl
    from ..rewards.safety_oracle import SafetyOracle
except ImportError:
    from utils.action_parser import (
        ActionType, ParsedAction, parse_action,
        is_terminal_action, build_system_prompt,
    )
    from utils.trajectory_utils import EpisodeState, compute_trajectory_stats
    from envs.failure_injection import FailureInjector, create_injector_from_episode, FailureType
    from envs.user_simulator import simulate_user_response
    from rewards.triage_reward import TriageRewardFunction, RewardBreakdown, get_reward_for_verl
    from rewards.safety_oracle import SafetyOracle

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration for the Triage Tool Environment."""
    max_turns: int = 12
    max_backtrack: int = 3
    max_replan: int = 1
    # Reward config
    reward_alpha: float = 0.5
    reward_format_weight: float = 0.3
    # Tools
    tools_registry_path: str = "data/tools_registry.json"


@dataclass
class StepResult:
    """Result of a single environment step."""
    observation: str           # What the agent sees next
    reward: float              # 0 for non-terminal, actual reward for terminal
    done: bool                 # Episode terminated?
    info: Dict[str, Any] = field(default_factory=dict)


class TriageToolEnv:
    """Multi-turn tool-use environment with 4-action triage policy.
    
    Lifecycle:
        1. reset(episode) → initial observation (system + user query)
        2. step(model_output) → StepResult (observation, reward, done, info)
        3. Repeat until done
        4. get_trajectory() → full episode data for logging
    """
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.reward_fn = TriageRewardFunction(
            alpha=config.reward_alpha,
            format_weight=config.reward_format_weight,
        )
        self.safety_oracle = SafetyOracle()
        
        # Load tools registry
        self.tools = self._load_tools(config.tools_registry_path)
        
        # Episode state
        self.state: Optional[EpisodeState] = None
        self.injector: Optional[FailureInjector] = None
        self.episode_data: Optional[Dict[str, Any]] = None
        
        # Tracking for reward
        self._tool_returns: List[str] = []
        self._format_valid: int = 0
        self._format_invalid: int = 0
    
    def _load_tools(self, path: str) -> List[Dict[str, Any]]:
        """Load tools from registry file."""
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("tools", [])
        except FileNotFoundError:
            logger.warning(f"Tools registry not found at {path}, using empty list")
            return []
    
    def reset(self, episode: Dict[str, Any]) -> str:
        """Reset environment for a new episode.
        
        Args:
            episode: Episode data dict (from train_demo.jsonl)
        
        Returns:
            Initial observation string (system prompt + user query)
        """
        self.episode_data = episode
        self.state = EpisodeState(
            episode_id=episode["id"],
            query=episode["query"],
        )
        self.injector = create_injector_from_episode(episode)
        self._tool_returns = []
        self._format_valid = 0
        self._format_invalid = 0
        
        # Filter tools to only those available for this episode
        available_tools = episode.get("tools_available", [])
        episode_tools = [t for t in self.tools if t["name"] in available_tools]
        if not episode_tools:
            episode_tools = self.tools  # Fallback to all tools
        
        # Build system prompt
        system_prompt = build_system_prompt(
            tools=episode_tools,
            max_backtrack=self.config.max_backtrack,
            max_replan=self.config.max_replan,
        )
        
        # Add system prompt and user query to state
        self.state.add_turn("system", system_prompt)
        self.state.add_turn("user", episode["query"])
        
        # Return the initial context as a single string for the model
        return self._format_initial_context(system_prompt, episode["query"])
    
    def step(self, model_output: str) -> StepResult:
        """Process one step of the agent's output.
        
        Args:
            model_output: Raw text output from the model
        
        Returns:
            StepResult with next observation, reward, and done flag
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if self.state.is_terminated:
            return StepResult(
                observation="Episode already terminated.",
                reward=0.0,
                done=True,
                info={"error": "episode_already_done"},
            )
        
        # Parse the action
        action = parse_action(model_output)
        
        # Record the assistant turn
        self.state.add_turn(
            "assistant",
            model_output,
            action_type=action.action_type.value,
            tool_name=action.tool_name,
            tool_params=action.tool_params,
        )
        
        # Dispatch based on action type
        if action.action_type == ActionType.TOOL_CALL:
            return self._handle_tool_call(action)
        elif action.action_type == ActionType.FINAL_ANSWER:
            return self._handle_final_answer(action)
        elif action.action_type == ActionType.BACKTRACK:
            return self._handle_backtrack(action)
        elif action.action_type == ActionType.REPLAN:
            return self._handle_replan(action)
        elif action.action_type == ActionType.REFUSE:
            return self._handle_refuse(action)
        elif action.action_type == ActionType.ASK_CLARIFY:
            return self._handle_ask_clarify(action)
        else:
            return self._handle_invalid(action)
    
    def _handle_tool_call(self, action: ParsedAction) -> StepResult:
        """Execute a tool call, possibly with failure injection."""
        self.state.total_tool_calls += 1
        
        # Validate tool name
        tool_names = [t["name"] for t in self.tools]
        if action.tool_name not in tool_names:
            self._format_invalid += 1
            error_msg = f"Tool '{action.tool_name}' not found. Available: {tool_names}"
            self.state.add_turn("tool", error_msg, tool_name=action.tool_name)
            self.state.failed_tool_calls += 1
            return self._check_max_turns(error_msg)
        
        self._format_valid += 1
        
        # Execute with possible failure injection
        response, was_failure = self.injector.intercept_tool_call(
            action.tool_name, action.tool_params or {}
        )
        
        self._tool_returns.append(response)
        
        if was_failure:
            self.state.failed_tool_calls += 1
        else:
            self.state.successful_tool_calls += 1
        
        # Record tool result
        self.state.add_turn(
            "tool",
            response,
            tool_name=action.tool_name,
            tool_result=response,
        )
        
        return self._check_max_turns(
            f"<tool_result>\n{response}\n</tool_result>"
        )
    
    def _handle_final_answer(self, action: ParsedAction) -> StepResult:
        """Handle final answer — terminate episode and compute reward."""
        self.state.is_terminated = True
        self.state.termination_reason = "final_answer"
        
        # Check task success
        task_success = self._check_answer(action.answer)
        
        # Compute reward
        breakdown = self.reward_fn.compute(
            task_success=task_success,
            agent_refused=False,
            backtrack_count=self.state.backtrack_count,
            replan_count=self.state.replan_count,
            safety_label=self.episode_data.get("safety_label", "safe"),
            tool_returns=self._tool_returns,
            format_valid_count=self._format_valid,
            format_invalid_count=self._format_invalid,
        )
        
        return StepResult(
            observation="Episode completed.",
            reward=get_reward_for_verl(breakdown),
            done=True,
            info={
                "reward_breakdown": breakdown.to_dict(),
                "task_success": task_success,
                "answer": action.answer,
                "trajectory_stats": compute_trajectory_stats(self.state),
            },
        )
    
    def _handle_backtrack(self, action: ParsedAction) -> StepResult:
        """Handle backtrack — roll back last step."""
        if self.state.backtrack_count >= self.config.max_backtrack:
            obs = (
                f"<system_note>Maximum backtrack limit ({self.config.max_backtrack}) reached. "
                f"Consider using <replan> for a different strategy, "
                f"<refuse> if the task is impossible, "
                f"or <final_answer> with your best answer.</system_note>"
            )
            self.state.add_turn("user", obs, metadata={"injected": True, "type": "backtrack_limit"})
            return self._check_max_turns(obs)
        
        # Roll back the last tool_call + tool_result
        rolled_call, rolled_result = self.state.rollback_last_step()
        
        if rolled_call is None:
            obs = "<system_note>Nothing to backtrack — no previous tool call found.</system_note>"
            self.state.add_turn("user", obs, metadata={"injected": True})
            return self._check_max_turns(obs)
        
        # Inject backtrack note
        self.state.inject_backtrack_note()
        
        obs = (
            f"<backtrack_note>Previous tool call to '{rolled_call.tool_name}' has been "
            f"removed. Reason: {action.backtrack_reason or 'unspecified'}. "
            f"Please try a different approach.</backtrack_note>"
        )
        
        return self._check_max_turns(obs)
    
    def _handle_replan(self, action: ParsedAction) -> StepResult:
        """Handle replan — keep history, inject replan note."""
        if self.state.replan_count >= self.config.max_replan:
            obs = (
                f"<system_note>Maximum replan limit ({self.config.max_replan}) reached. "
                f"Consider <refuse> if the task seems impossible, "
                f"or <final_answer> with your best answer.</system_note>"
            )
            self.state.add_turn("user", obs, metadata={"injected": True, "type": "replan_limit"})
            return self._check_max_turns(obs)
        
        # Inject replan note (keeps all history)
        self.state.inject_replan_note()
        
        obs = (
            f"<replan_note>Strategy reset. Previous approaches have failed. "
            f"Reason: {action.replan_reason or 'unspecified'}. "
            f"All history is preserved for reference. "
            f"Please devise a completely different approach.</replan_note>"
        )
        
        return self._check_max_turns(obs)
    
    def _handle_refuse(self, action: ParsedAction) -> StepResult:
        """Handle refuse — terminate episode, compute safety reward."""
        self.state.is_terminated = True
        self.state.termination_reason = "refuse"
        
        # Compute reward
        breakdown = self.reward_fn.compute(
            task_success=False,
            agent_refused=True,
            backtrack_count=self.state.backtrack_count,
            replan_count=self.state.replan_count,
            safety_label=self.episode_data.get("safety_label", "safe"),
            tool_returns=self._tool_returns,
            format_valid_count=self._format_valid,
            format_invalid_count=self._format_invalid,
        )
        
        return StepResult(
            observation="Episode terminated by agent refusal.",
            reward=get_reward_for_verl(breakdown),
            done=True,
            info={
                "reward_breakdown": breakdown.to_dict(),
                "refuse_reason": action.refuse_reason,
                "trajectory_stats": compute_trajectory_stats(self.state),
            },
        )
    
    def _handle_ask_clarify(self, action: ParsedAction) -> StepResult:
        """Handle ask_clarify — simulate user response."""
        user_response = simulate_user_response(
            clarify_question=action.clarify_question or "",
            original_query=self.state.query,
            turn_number=self.state.current_turn,
        )
        
        self.state.add_turn("user", user_response)
        
        return self._check_max_turns(user_response)
    
    def _handle_invalid(self, action: ParsedAction) -> StepResult:
        """Handle invalid/unparseable action."""
        self._format_invalid += 1
        obs = (
            "<system_note>Invalid action format. Please use one of: "
            "<tool_call>, <final_answer>, <backtrack>, <replan>, <refuse>, <ask_clarify>. "
            "Check the format requirements in the system prompt.</system_note>"
        )
        self.state.add_turn("user", obs, metadata={"injected": True, "type": "invalid_format"})
        return self._check_max_turns(obs)
    
    def _check_max_turns(self, observation: str) -> StepResult:
        """Check if max turns reached, return appropriate StepResult."""
        if self.state.current_turn >= self.config.max_turns:
            self.state.is_terminated = True
            self.state.termination_reason = "max_turns"
            
            # Compute reward (task failed due to timeout)
            breakdown = self.reward_fn.compute(
                task_success=False,
                agent_refused=False,
                backtrack_count=self.state.backtrack_count,
                replan_count=self.state.replan_count,
                safety_label=self.episode_data.get("safety_label", "safe"),
                tool_returns=self._tool_returns,
                format_valid_count=self._format_valid,
                format_invalid_count=self._format_invalid,
            )
            
            return StepResult(
                observation=observation + "\n<system_note>Maximum turns reached. Episode terminated.</system_note>",
                reward=get_reward_for_verl(breakdown),
                done=True,
                info={
                    "reward_breakdown": breakdown.to_dict(),
                    "trajectory_stats": compute_trajectory_stats(self.state),
                    "terminated_by": "max_turns",
                },
            )
        
        return StepResult(
            observation=observation,
            reward=0.0,
            done=False,
        )
    
    def _check_answer(self, answer: Optional[str]) -> bool:
        """Check if the agent's answer matches the gold answer.
        
        Uses exact match (case-insensitive, stripped) for demo.
        In production, use EM + F1 scoring.
        """
        if answer is None:
            return False
        
        gold = self.episode_data.get("gold_answer")
        if gold is None:
            # For harmful/impossible tasks, there's no gold answer
            # Task "success" doesn't apply
            return False
        
        # Simple containment check for demo
        answer_lower = answer.lower().strip()
        gold_lower = gold.lower().strip()
        
        # Check if key parts of gold answer appear in agent's answer
        # This is a rough heuristic; production would use F1
        gold_tokens = set(gold_lower.split())
        answer_tokens = set(answer_lower.split())
        
        if len(gold_tokens) == 0:
            return False
        
        overlap = gold_tokens.intersection(answer_tokens)
        f1_approx = len(overlap) / max(len(gold_tokens), 1)
        
        return f1_approx > 0.5  # Generous threshold for demo
    
    def _format_initial_context(self, system_prompt: str, query: str) -> str:
        """Format the initial context string."""
        return f"{system_prompt}\n\n---\nUser Query: {query}\n---\n\nPlease begin solving this task."
    
    def get_trajectory(self) -> Dict[str, Any]:
        """Get the full trajectory data for logging."""
        if self.state is None:
            return {}
        return self.state.to_dict()
    
    def get_messages_for_model(self) -> List[Dict[str, str]]:
        """Get the current conversation in chat format for the model."""
        if self.state is None:
            return []
        return self.state.get_context_messages()
