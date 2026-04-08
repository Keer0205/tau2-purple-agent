import os
import logging
import anthropic

print("PRINT_MARKER_AGENT_MODULE_LOADED_DEBUG_V7", flush=True)

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
4. If you can verify or progress the request using the available conversation context, do that immediately.
5. Ask a clarifying question only when absolutely necessary.
6. Ask for only one missing detail at a time.
7. Do not ask for information the user already provided.
8. If the user gives enough information to continue, continue directly in the next response.
9. Do not output XML, HTML, tool tags, markdown code fences, or JSON.
10. Return only plain natural-language text.
11. Do not mention internal tools unless they are actually available and used.
12. If no tools are available, do not pretend to call tools.
13. Do not include function-call syntax or structured markup in your reply.
14. If identity verification is required, ask only for the minimum necessary detail.
15. After identity is verified, continue the task directly instead of asking unnecessary extra confirmation.
16. Prefer completing or advancing the airline task over giving general explanations.
17. If the user asks to cancel, modify, refund, add bags, or change seats, move the task forward as far as possible in the same turn.
18. Keep most responses short and practical.

Your job is to help the user complete the airline task safely and correctly in plain text.
"""


def _sanitize_text(text: str) -> str:
    if not text:
        return ""

    cleanup_markers = [
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<function_calls>",
        "</function_calls>",
        "<invoke>",
        "</invoke>",
        "```xml",
        "```json",
        "```python",
        "```",
    ]
    for marker in cleanup_markers:
        text = text.replace(marker, "")

    return text.strip()


def run_agent(task: str, tools: list, conversation_history: list) -> tuple:
    print("PRINT_MARKER_RUN_AGENT_ENTERED_DEBUG_V7", flush=True)

    client = get_client()
    model = get_model()
    messages = conversation_history.copy() if conversation_history else []

    logger.info(f"run_agent called with task: {task[:300] if task else ''}")
    logger.info(f"run_agent tools count: {len(tools) if tools else 0}")
    logger.info(f"run_agent history length: {len(messages)}")

    print(f"PRINT_MARKER_RUN_AGENT_TOOLS_COUNT={len(tools) if tools else 0}", flush=True)
    print(f"PRINT_MARKER_RUN_AGENT_HISTORY_LEN={len(messages)}", flush=True)

    if not messages:
        messages = [{"role": "user", "content": task}]
    elif task and messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": task})

    if tools:
        logger.info(f"Tools available in context: count={len(tools)}")
        print(f"PRINT_MARKER_TOOLS_AVAILABLE count={len(tools)}", flush=True)
    else:
        logger.info("No tools available in context")
        print("PRINT_MARKER_NO_TOOLS_AVAILABLE", flush=True)

    kwargs = {
        "model": model,
        "max_tokens": 900,
        "temperature": 0,
        "system": SYSTEM_PREFIX,
        "messages": messages,
    }

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

    text_response = _sanitize_text("".join(text_parts))

    logger.info(f"Returning final text response length={len(text_response)}")
    print("PRINT_MARKER_AGENT_RETURN_FINAL_TEXT_V7", flush=True)

    messages.append({"role": "assistant", "content": text_response})
    return text_response, messages
