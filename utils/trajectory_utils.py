"""
Trajectory utilities for managing multi-turn tool-use trajectories.

Handles state rollback (backtrack), history preservation (replan),
and trajectory recording for reward computation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import copy
import json


@dataclass
class TurnRecord:
    """Record of a single turn in the trajectory."""
    turn_id: int
    role: str  # "assistant" | "tool" | "user" | "system"
    content: str
    action_type: Optional[str] = None  # ActionType.value
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    is_rolled_back: bool = False  # Marked True if backtracked
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeState:
    """Full state of an episode, supporting rollback and replan."""
    episode_id: str
    query: str
    turns: List[TurnRecord] = field(default_factory=list)
    backtrack_count: int = 0
    replan_count: int = 0
    is_terminated: bool = False
    termination_reason: Optional[str] = None  # "final_answer" | "refuse" | "max_turns"
    
    # Counters for reward computation
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    recovery_attempts: int = 0
    
    @property
    def current_turn(self) -> int:
        return len([t for t in self.turns if not t.is_rolled_back])
    
    def get_active_turns(self) -> List[TurnRecord]:
        """Get turns that haven't been rolled back."""
        return [t for t in self.turns if not t.is_rolled_back]
    
    def add_turn(self, role: str, content: str, **kwargs) -> TurnRecord:
        """Add a new turn to the trajectory."""
        turn = TurnRecord(
            turn_id=len(self.turns),
            role=role,
            content=content,
            **kwargs,
        )
        self.turns.append(turn)
        return turn
    
    def rollback_last_step(self) -> Tuple[Optional[TurnRecord], Optional[TurnRecord]]:
        """Roll back the last tool_call + tool_result pair.
        
        Returns:
            Tuple of (rolled_back_call, rolled_back_result) or (None, None)
        """
        active = self.get_active_turns()
        rolled_back = []
        
        # Find and mark the last tool_call + tool_result as rolled back
        for turn in reversed(active):
            if turn.role == "tool" and not turn.is_rolled_back:
                turn.is_rolled_back = True
                rolled_back.append(turn)
            elif turn.role == "assistant" and turn.action_type == "tool_call" and not turn.is_rolled_back:
                turn.is_rolled_back = True
                rolled_back.append(turn)
                break
        
        self.backtrack_count += 1
        self.recovery_attempts += 1
        
        if len(rolled_back) == 2:
            return rolled_back[1], rolled_back[0]  # (call, result)
        return None, None
    
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Build the conversation context from active (non-rolled-back) turns.
        
        This is what gets fed to the model as conversation history.
        """
        messages = []
        for turn in self.get_active_turns():
            if turn.role == "system":
                messages.append({"role": "system", "content": turn.content})
            elif turn.role == "assistant":
                messages.append({"role": "assistant", "content": turn.content})
            elif turn.role == "tool":
                # Tool results are formatted as user messages with tool output
                messages.append({
                    "role": "user",
                    "content": f"<tool_result>\n{turn.content}\n</tool_result>"
                })
            elif turn.role == "user":
                messages.append({"role": "user", "content": turn.content})
        return messages
    
    def inject_backtrack_note(self) -> None:
        """Inject a note into context after backtrack."""
        note = (
            "<backtrack_note>The previous tool call failed or returned "
            "invalid results. It has been removed from context. "
            "Please try a different approach — use a different tool, "
            "different parameters, or a different strategy.</backtrack_note>"
        )
        self.add_turn("user", note, metadata={"injected": True, "type": "backtrack_note"})
    
    def inject_replan_note(self) -> None:
        """Inject a note into context for replan (keep all history)."""
        note = (
            "<replan_note>Multiple recovery attempts have failed. "
            "Your previous strategy is not working. Please reconsider "
            "the overall approach — think about whether there's a "
            "completely different way to accomplish this task. "
            "All previous history is preserved for reference.</replan_note>"
        )
        self.add_turn("user", note, metadata={"injected": True, "type": "replan_note"})
        self.replan_count += 1
        self.recovery_attempts += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize episode state for logging."""
        return {
            "episode_id": self.episode_id,
            "query": self.query,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "role": t.role,
                    "content": t.content[:200] + "..." if len(t.content) > 200 else t.content,
                    "action_type": t.action_type,
                    "tool_name": t.tool_name,
                    "is_rolled_back": t.is_rolled_back,
                }
                for t in self.turns
            ],
            "backtrack_count": self.backtrack_count,
            "replan_count": self.replan_count,
            "is_terminated": self.is_terminated,
            "termination_reason": self.termination_reason,
            "total_tool_calls": self.total_tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "recovery_attempts": self.recovery_attempts,
        }


def compute_trajectory_stats(state: EpisodeState) -> Dict[str, Any]:
    """Compute statistics from an episode trajectory for analysis."""
    active_turns = state.get_active_turns()
    
    # Find positions of recovery actions
    recovery_positions = []
    for i, turn in enumerate(active_turns):
        if turn.action_type in ("backtrack", "replan", "refuse"):
            recovery_positions.append(i / max(len(active_turns), 1))
    
    return {
        "total_turns": len(state.turns),
        "active_turns": len(active_turns),
        "rolled_back_turns": len(state.turns) - len(active_turns),
        "backtrack_count": state.backtrack_count,
        "replan_count": state.replan_count,
        "recovery_attempts": state.recovery_attempts,
        "recovery_positions": recovery_positions,  # Normalized [0,1] positions
        "avg_recovery_position": (
            sum(recovery_positions) / len(recovery_positions)
            if recovery_positions else None
        ),
        "terminated": state.is_terminated,
        "termination_reason": state.termination_reason,
    }
