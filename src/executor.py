import os
import anthropic

def get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PREFIX = """You are an airline customer service agent. Help customers with their requests professionally and efficiently."""

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
