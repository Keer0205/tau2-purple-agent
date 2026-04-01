import os
import json
import anthropic
from typing import Any

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PREFIX = """You are an expert customer service agent for airline, retail, and telecom domains. You will receive a domain policy and tools.

CRITICAL RULES:
1. Read the domain policy carefully - follow ALL rules EXACTLY
2. Use ONE tool at a time - never call multiple tools at once
3. ALWAYS verify customer identity before making any changes
4. Never make up information - only use data from tool results
5. Complete the task FULLY before ending the conversation
6. If policy says ask first - ALWAYS ask before acting
7. Be concise, professional and helpful
8. Confirm every completed action to the user
9. If a tool call fails, retry with corrected parameters
10. Track what actions you have taken to avoid duplicates
11. When the user confirms an action, execute it immediately
12. Always summarize what was done at the end of the conversation"""

def run_agent(task: str, tools: list, conversation_history: list) -> tuple:
    messages = conversation_history.copy()
    if not messages:
        messages = [{"role": "user", "content": task}]
    elif task and (not messages or messages[-1].get("role") != "user"):
        messages.append({"role": "user", "content": task})

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
                    "input_schema": tool.get("parameters", tool.get("input_schema", {"type": "object", "properties": {}}))
                })

    for iteration in range(20):
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "system": SYSTEM_PREFIX,
            "messages": messages
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = client.messages.create(**kwargs)
        assistant_content = []
        text_response = ""
        tool_uses = []

        for block in response.content:
            if hasattr(block, "text"):
                text_response += block.text
                assistant_content.append({"type": "text", "text": block.text})
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_uses.append(block)
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn" or not tool_uses:
            return text_response, messages

        if tool_uses:
            tool_block = tool_uses[0]
            tool_call_response = json.dumps({
                "type": "function",
                "function": {
                    "name": tool_block.name,
                    "arguments": json.dumps(tool_block.input)
                }
            })
            return tool_call_response, messages

    return "I was unable to complete the task. Please try again.", messages
