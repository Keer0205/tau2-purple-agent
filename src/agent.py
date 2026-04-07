import os
import logging
import anthropic

print("PRINT_MARKER_AGENT_MODULE_LOADED_DEBUG_V4", flush=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_client():
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AMBER_CONFIG_AGENT_ANTHROPIC_API_KEY")
    )
    logger.info(f"API key found: {bool(api_key)}")
    print(f"PRINT_MARKER_API_KEY_FOUND={bool(api_key)}", flush=True)
    return anthropic.Anthropic(api_key=api_key)


def get_model():
    model = (
        os.environ.get("AGENT_LLM")
        or os.environ.get("AMBER_CONFIG_AGENT_AGENT_LLM")
        or "anthropic/claude-3-5-sonnet-20241022"
    )
    logger.info(f"Using model: {model}")
    print(f"PRINT_MARKER_AGENT_MODEL={model}", flush=True)
    return model


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
    print("PRINT_MARKER_RUN_AGENT_ENTERED_DEBUG_V4", flush=True)

    client = get_client()
    model = get_model()
    messages = conversation_history.copy()

    logger.info(f"run_agent called with task: {task[:300] if task else ''}")
    logger.info(f"run_agent tools count: {len(tools) if tools else 0}")
    logger.info(f"run_agent history length: {len(messages)}")

    print(f"PRINT_MARKER_RUN_AGENT_TOOLS_COUNT={len(tools) if tools else 0}", flush=True)
    print(f"PRINT_MARKER_RUN_AGENT_HISTORY_LEN={len(messages)}", flush=True)

    if not messages:
        messages = [{"role": "user", "content": task}]
    elif task and messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": task})

    max_iterations = 10
    text_response = ""

    for iteration in range(max_iterations):
        print(f"PRINT_MARKER_AGENT_LOOP_ITERATION={iteration}", flush=True)

        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "system": SYSTEM_PREFIX,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools
            logger.info(f"Passing tools to Anthropic: count={len(tools)}")
            print(f"PRINT_MARKER_TOOLS_ATTACHED_TO_MODEL count={len(tools)}", flush=True)
        else:
            logger.warning("No tools passed to Anthropic")
            print("PRINT_MARKER_NO_TOOLS_ATTACHED", flush=True)

        response = client.messages.create(**kwargs)
        logger.info(
            f"Iteration {iteration}: stop_reason={response.stop_reason}, blocks={len(response.content)}"
        )
        print(
            f"PRINT_MARKER_AGENT_RESPONSE stop_reason={response.stop_reason} blocks={len(response.content)}",
            flush=True,
        )

        text_parts = []
        tool_use_blocks = []

        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_use_blocks.append(block)

        text_response = "".join(text_parts)
        messages.append({"role": "assistant", "content": response.content})

        print(
            f"PRINT_MARKER_TOOL_USE_BLOCKS count={len(tool_use_blocks)}",
            flush=True,
        )

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            logger.info("Returning final text response without more tool calls")
            print("PRINT_MARKER_AGENT_RETURN_END_TURN", flush=True)
            return text_response, messages

        tool_results = []
        for tool_block in tool_use_blocks:
            logger.info(f"Tool call: {tool_block.name}({tool_block.input})")
            print(
                f"PRINT_MARKER_TOOL_CALL name={tool_block.name} input={tool_block.input}",
                flush=True,
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": f"Tool '{tool_block.name}' dispatched with input: {tool_block.input}",
            })

        messages.append({"role": "user", "content": tool_results})

    logger.warning("Max iterations reached in agent loop")
    print("PRINT_MARKER_AGENT_MAX_ITERATIONS_REACHED", flush=True)
    return text_response, messages
