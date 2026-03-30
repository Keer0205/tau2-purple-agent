import os
import anthropic
from typing import Any

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a helpful customer service agent for airline, retail, and telecom companies.

Your goals:
- Be polite, accurate, and efficient
- Use the tools provided to complete customer requests
- Always verify information before making changes
- Follow company policies strictly
- Ask clarifying questions when needed
- Never make up information

When handling requests:
1. Understand the customer's issue
2. Use available tools to look up information
3. Take appropriate actions using tools
4. Confirm actions with the customer
5. Ensure the customer is satisfied"""

def run_agent(task: str, tools: list[dict], conversation_history: list[dict]) -> tuple[str, list[dict]]:
    messages = conversation_history.copy()
    
    if not messages or messages[0].get("role") != "user":
        messages = [{"role": "user", "content": task}]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools if tools else [],
            messages=messages
        )
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.content})
        
        # If no tool use, return the text response
        if response.stop_reason == "end_turn":
            text = next((b.text for b in response.content if hasattr(b, "text")), "")
            return text, messages
        
        # Handle tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool {block.name} called with {block.input}"
                    })
            
            messages.append({"role": "user", "content": tool_results})
        else:
            text = next((b.text for b in response.content if hasattr(b, "text")), "")
            return text, messages
