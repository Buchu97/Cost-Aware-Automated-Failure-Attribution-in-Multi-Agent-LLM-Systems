"""Build the All-at-Once failure attribution prompt for the Who&When benchmark."""
from src.agent_names import allowed_agents


def build_attribution_prompt(trace: dict) -> str:
    """Construct the All-at-Once prompt for a single failure trace.

    The prompt matches the original Who&When paper's format but adds a
    confidence-score request so the model self-reports its certainty.

    Args:
        trace: A trace dict as returned by data_loader.load_traces().

    Returns:
        The complete prompt string to send to the LLM.
    """
    # Algorithm-Generated traces use 'name'; Hand-Crafted use 'role'
    agent_key = "role" if trace["is_handcrafted"] else "name"

    # Build numbered conversation string
    steps: list[str] = []
    for i, entry in enumerate(trace["messages"]):
        agent = entry.get(agent_key, "Unknown")
        content = entry.get("content", "")
        steps.append(f"Step {i} - {agent}: {content}")
    chat_content = "\n".join(steps)
    names = ', '.join(sorted(allowed_agents(trace)))

    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation "
        "history when solving a real world problem.\n\n"
        f"The problem is: {trace['question']}\n"
        f"The correct answer is: {trace['ground_truth']}\n\n"
        "The agents failed to solve this problem correctly. "
        "Identify which agent made the critical error, at which step, and explain why.\n\n"
        f"Agent names: {names}.\n"
        "Use an exact agent name from this list. Orchestrator role suffixes in parentheses "
        "describe its message type or recipient; the agent name is Orchestrator.\n\n"
        f"Here is the conversation:\n\n{chat_content}\n\n"
        "Based on this conversation, predict the following:\n"
        "1. Agent Name: The exact name of the agent most directly responsible "
        "for the failure.\n"
        "2. Step Number: The 0-indexed step number where that agent first made "
        "the critical mistake.\n"
        "3. Confidence: Your confidence in this attribution (integer from 0 to 100).\n"
        "4. Reason: A brief explanation.\n\n"
        "Respond in EXACTLY this format (no extra text):\n"
        "Agent Name: <name>\n"
        "Step Number: <number>\n"
        "Confidence: <0-100>\n"
        "Reason: <explanation>\n"
    )
    return prompt
