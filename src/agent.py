import os
import anthropic

def get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PREFIX = """You are an airline customer service agent. Your job is to help customers with their requests by taking concrete actions.

IMPORTANT BEHAVIOR:
- When a customer contacts you, immediately ask for their name and booking reference to verify their identity
- Once verified, take the specific action they request (cancel flight, change seat, request refund, etc.)
- Always confirm what action you have taken
- Be direct and efficient - complete the task fully
- Never just say "How can I help" - always move the conversation forward
- If the customer wants to cancel: verify identity, confirm cancellation, process it
- If the customer wants a refund: verify eligibility, process the refund
- If the customer wants to change seat/flight: verify identity, make the change, confirm

Always complete the full task in as few turns as possible."""

def run_agent(task: str, tools: list, conversation_history: list) -> tuple:
    messages = conversation_history.copy()
    
    if not messages:
        messages = [{"role": "user", "content": task}]
    elif task and messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": task})

    response = get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PREFIX,
        messages=messages
    )

    text_response = ""
    for block in response.content:
        if hasattr(block, "text"):
            text_response += block.text

    messages.append({"role": "assistant", "content": text_response})
    return text_response, messages
