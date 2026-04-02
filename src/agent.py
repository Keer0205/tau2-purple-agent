import os
import anthropic

def get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PREFIX = """You are an expert customer service agent for airline, retail, and telecom domains. You will receive a domain policy and tools.

CRITICAL RULES:
1. Read the domain policy carefully - follow ALL rules EXACTLY
2. ALWAYS verify customer identity before making any changes
3. Never make up information - only use data from tool results
4. Complete the task FULLY before ending the conversation
5. Be concise, professional and helpful
6. Confirm every completed action to the user
7. Always summarize what was done at the end of the conversation"""

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
