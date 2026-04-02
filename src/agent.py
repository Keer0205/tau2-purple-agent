import os
import logging
import anthropic

logger = logging.getLogger(__name__)

def get_client():
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY") or
        os.environ.get("AMBER_CONFIG_AGENT_ANTHROPIC_API_KEY")
    )
    logger.info(f"API key found: {bool(api_key)}")
    return anthropic.Anthropic(api_key=api_key)

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
    client = get_client()
    messages = conversation_history.copy()

    if not messages:
        messages = [{"role": "user", "content": task}]
    elif task and messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": task})

    max_iterations = 10
    for iteration in range(max_iterations):
        kwargs = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
            "system": SYSTEM_PREFIX,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)
        logger.info(f"Iteration {iteration}: stop_reason={response.stop_reason}, blocks={len(response.content)}")

        text_parts = []
        tool_use_blocks = []

        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        text_response = "".join(text_parts)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            return text_response, messages

        tool_results = []
        for tool_block in tool_use_blocks:
            logger.info(f"Tool call: {tool_block.name}({tool_block.input})")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": f"Tool '{tool_block.name}' dispatched with input: {tool_block.input}",
            })

        messages.append({"role": "user", "content": tool_results})

    logger.warning("Max iterations reached in agent loop")
    return text_response, messages
