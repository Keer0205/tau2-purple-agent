import os
import logging
import anthropic

print("PRINT_MARKER_AGENT_MODULE_LOADED_DEBUG_V6", flush=True)

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


SYSTEM_PREFIX = """You are an airline support agent operating inside a benchmark environment.

Your goal is to complete or advance the user's airline request as efficiently as possible.

CRITICAL RULES:
1. Handle only airline-support tasks.
2. Be concise, direct, accurate, and practical.
3. Never invent facts, bookings, policies, balances, account details, or completed actions.
4. Use the conversation context carefully. Do not ask for information the user already provided.
5. Do not ask for an ID, username, booking reference, or account number unless it is truly required to continue.
6. Ask a clarification question only when a missing detail is genuinely necessary.
7. Ask at most one focused clarification at a time.
8. If the user provides the needed detail, continue the task immediately in the next response.
9. Prefer progressing the task over giving general explanations.
10. For cancellations, changes, refunds, baggage, seats, check-in, booking lookup, and account lookup, move the task forward as far as possible in the same turn.
11. If identity verification is required, ask only for the minimum necessary detail.
12. After identity is verified, do not repeat verification or ask for unnecessary extra confirmation.
13. If you cannot actually execute an action, do not claim it has been completed.
14. When execution is not possible, give the most concrete next step or best available outcome allowed by the conversation context.
15. Do not use generic customer-service filler such as "I'd be happy to help", "Let me check", "Please let me know", or long apologies.
16. Do not output XML, HTML, JSON, markdown, code fences, tool tags, or function-call syntax.
17. Return only plain natural-language text.
18. If no tools are available, do not pretend to use tools.
19. Do not mention internal systems, internal tools, or hidden policies unless directly relevant.
20. Keep most responses to 1-3 short sentences.
21. If you can answer directly, do so.
22. If you can partially complete the task, do so before asking for more information.

STYLE EXAMPLES:
Good: "What is your booking reference?"
Good: "Please share the travel date."
Good: "This fare appears non-refundable. I can still help with change options."
Bad: "I'd be happy to help you with that today."
Bad: "I have processed your refund." 
Bad: "Let me check that for you."
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
    print("PRINT_MARKER_RUN_AGENT_ENTERED_DEBUG_V6", flush=True)

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
        "max_tokens": 700,
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
    print("PRINT_MARKER_AGENT_RETURN_FINAL_TEXT_V6", flush=True)

    messages.append({"role": "assistant", "content": text_response})
    return text_response, messages
