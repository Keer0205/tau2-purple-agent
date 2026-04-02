import os
import json
import anthropic

def get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PREFIX = """You are an airline customer service agent with access to tools. Use the tools to complete customer requests.

CRITICAL RULES:
1. ALWAYS use tools to look up information and make changes - never guess
2. Verify customer identity first using tools
3. Complete the full task using tools before ending
4. Be concise and professional
5. After each tool call, use the result to continue helping"""

def run_agent(task: str, tools: list, conversation_history: list) -> tuple:
    messages = conversation_history.copy()

    if not messages:
        messages = [{"role": "user", "content": task}]
    elif task and messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": task})

    # Convert tools to Anthropic format
    anthropic_tools = []
    for tool in tools:
        if isinstance(tool, dict):
            if "function" in tool:
                func = tool["function"]
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
            elif "name" in tool:
                anthropic_tools.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema", tool.get("parameters", {"type": "object", "properties": {}}))
                })

    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": SYSTEM_PREFIX,
        "messages": messages
    }
    if anthropic_tools:
        kwargs["tools"] = anthropic_tools

    response = get_client().messages.create(**kwargs)

    text_response = ""
    for block in response.content:
        if hasattr(block, "text"):
            text_response += block.text

    messages.append({"role": "assistant", "content": response.content if anthropic_tools else text_response})
    return text_response, messages
