"""
User Simulator

When the agent issues <ask_clarify>, this module generates a simulated
user response. For the demo, we use template-based responses.
In production, this would use an LLM (e.g., GPT-4o-mini) to simulate.
"""

import random
from typing import Optional


# Template responses for common clarification scenarios
CLARIFICATION_TEMPLATES = {
    "which_database": [
        "Use the main production database.",
        "The customer database, please.",
        "Check the analytics database.",
    ],
    "which_format": [
        "Please provide the results in a table format.",
        "Just a summary is fine.",
        "Give me the raw data.",
    ],
    "time_range": [
        "For the last 30 days.",
        "From this year only.",
        "Just the most recent data available.",
    ],
    "default": [
        "Yes, please go ahead with the default option.",
        "Use your best judgment.",
        "Just give me whatever you think is most relevant.",
        "The standard approach should be fine.",
    ],
}


def simulate_user_response(
    clarify_question: str,
    original_query: str,
    turn_number: int,
) -> str:
    """Generate a simulated user response to a clarification question.
    
    Args:
        clarify_question: The agent's clarification question
        original_query: The original user query
        turn_number: Current turn number (later turns → more likely to be impatient)
    
    Returns:
        Simulated user response string
    """
    question_lower = clarify_question.lower()
    
    # Match to templates based on question content
    if any(kw in question_lower for kw in ["database", "which db", "which table"]):
        responses = CLARIFICATION_TEMPLATES["which_database"]
    elif any(kw in question_lower for kw in ["format", "how would you like", "preference"]):
        responses = CLARIFICATION_TEMPLATES["which_format"]
    elif any(kw in question_lower for kw in ["time", "date", "period", "range", "when"]):
        responses = CLARIFICATION_TEMPLATES["time_range"]
    else:
        responses = CLARIFICATION_TEMPLATES["default"]
    
    response = random.choice(responses)
    
    # If late in the conversation, user might be impatient
    if turn_number > 6:
        response += " Please try to complete the task quickly."
    
    return response
