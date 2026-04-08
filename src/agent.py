import os
import logging
import anthropic

print("PRINT_MARKER_AGENT_MODULE_LOADED_DEBUG_V5", flush=True)

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
        or "claude-sonnet-4-6"
    )
    logger.info(f"Using model: {model}")
    print(f"PRINT_MARKER_AGENT_MODEL={model}", flush=True)
    return model


SYSTEM_PREFIX = """You are an airline customer service agent.

Follow these rules exactly:
1. Focus only on airline support tasks.
2. Be concise, helpful, and accurate.
3. Never invent facts.
4. If you can verify or progress the request using the policy and available conversation context, do that immediately. Ask a clarifying question only when absolutely necessary.
5. Do not output XML, HTML, tool_call tags, tool_response tags, markdown code fences, or JSON.
6. Return only plain natural-language text for the assistant reply.
7. Do not mention internal tools unless they are actually available and used.
8. If no tools are available, do not pretend to call tools.
9. Do not include any function-call syntax or structured markup in your reply.
10. If identity verification is required by policy, ask only for the minimum necessary verification details.
11. Do not stop after a single verification question if the user reply gives enough information to continue.
12. After identity is verified, continue the task directly instead of asking for unnecessary extra confirmation.
13. Prefer completing the user’s requested airline task over asking general follow-up questions.
14. If the user asks to cancel, modify, refund, add bags, or change seats, move the task forward as far as policy allows in the same turn.

Your job is to help the user complete the airline task safely and correctly in plain text.
"""


def run_agent(task: str, tools: list, conversation_history: list) -> tuple:
    print("PRINT_MARKER_RUN_AGENT_ENTERED_DEBUG_V5", flush=True)

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

    kwargs = {
        "model": model,
        "max_tokens": 1200,
        "temperature": 0,
        "system": SYSTEM_PREFIX,
        "messages": messages,
    }

    if tools:
        logger.info(f"Tools available in context: count={len(tools)}")
        print(f"PRINT_MARKER_TOOLS_AVAILABLE count={len(tools)}", flush=True)
    else:
        logger.info("No tools available in context")
        print("PRINT_MARKER_NO_TOOLS_AVAILABLE", flush=True)

    response = client.messages.create(**kwargs)

    logger.info(
        f"Single call response: stop_reason={response.stop_reason}, blocks={len(response.content)}"
    )
    print(
        f"PRINT_MARKER_AGENT_RESPONSE stop_reason={response.stop_reason} blocks={len(response.content)}",
        flush=True,
    )

    text_parts = []
    for block in response.content:
        if hasattr(block, "text") and block.text:
            text_parts.append(block.text)

    text_response = "".join(text_parts).strip()

    # Safety cleanup: remove common accidental tool/function markup if the model emits it anyway.
    cleanup_markers = [
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<function_calls>",
        "</function_calls>",
        "<invoke>",
        "</invoke>",
    ]
    for marker in cleanup_markers:
        text_response = text_response.replace(marker, "")

    logger.info(f"Returning final text response length={len(text_response)}")
    print("PRINT_MARKER_AGENT_RETURN_FINAL_TEXT_V5", flush=True)

    messages.append({"role": "assistant", "content": text_response})
    return text_response, messages
