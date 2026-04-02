import os
import logging
import anthropic

logger = logging.getLogger(__name__)

def get_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PREFIX = """You are an expert customer service agent for airline, retail, and telecom domains.

CRITICAL RULES:
1. Read the domain policy carefully - follow ALL rules EXACTLY
2. ALWAYS verify customer identity before making any changes
3. Never make up information - only use data from tool results
4. Complete the task FULLY before ending the conversation
5. Be concise, professional and helpful
6. Confirm every completed action to the user
7. Always summarize what was done at the end

IMPORTANT - TOOL USE:
- You have access to tools. USE THEM to look up data, make changes, and complete tasks.
- Do NOT ask the user for information you can retrieve via tools.
- Always call the relevant tool first, then respond based on the result.
- If a tool call fails, explain what happened and try an alternative approach.

TASK COMPLETION:
- When you receive a task, identify what needs to be done and do it immediately.
- Do not greet the user and wait - start working on the task right away.
- Keep going until the task is fully resolved."""


def run_agent(task: str, tools: list, conversation_history: list) -> tuple:
    """
    Run the agent for one turn.
    Handles multi-step tool use in a loop until the model stops calling tools.
    Returns (response_text, updated_history).
    """
    client = get_client()
    messages = conversation_history.copy()

    # Add the new user message if needed
    if not messages:
        messages = [{"role": "user", "content": task}]
    elif task and messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": task})

    # Agentic loop — keep going while the model wants to use tools
    max_iterations = 10
    for iteration in range(max_iterations):
        kwargs = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
            "system": SYSTEM_PREFIX,
            "messages": messages,
        }

        # Only pass tools if we have them
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)
        logger.info(f"Iteration {iteration}: stop_reason={response.stop_reason}, blocks={len(response.content)}")

        # Collect text and tool calls from this response
        text_parts = []
        tool_use_blocks = []

        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        text_response = "".join(text_parts)

        # Append the assistant turn (with all content blocks)
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls or stop_reason is end_turn, we're done
        if response.stop_reason == "end_turn" or not tool_use_blocks:
            return text_response, messages

        # Execute each tool call and collect results
        tool_results = []
        for tool_block in tool_use_blocks:
            logger.info(f"Tool call: {tool_block.name}({tool_block.input})")
            # The actual tool execution happens via MCP on the green agent side.
            # We submit the tool_use block; results come back in the next user message
            # from the A2A/MCP layer. For now, record a placeholder so the loop continues.
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": f"Tool '{tool_block.name}' dispatched with input: {tool_block.input}",
            })

        # Add tool results as a user turn so the model can continue
        messages.append({"role": "user", "content": tool_results})

    logger.warning("Max iterations reached in agent loop")
    return text_response, messages
