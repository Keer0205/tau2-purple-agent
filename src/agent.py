import json
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


SYSTEM_PREFIX = """You are an airline customer service agent.

You must output exactly one JSON object per turn.

Output format:
{
  "name": "<tool_or_respond_name>",
  "arguments": { ... }
}

Rules:
1. Output valid JSON only. No markdown. No code fences. No extra commentary.
2. Use exactly one action per turn.
3. If you have enough information and a tool is available, prefer using the correct tool.
4. If a tool is not available, use:
   {"name":"respond","arguments":{"content":"..."}}
5. For booking/modifying/canceling/updating actions, follow policy carefully.
6. Before any database-changing action, list the action details and obtain explicit user confirmation when policy requires it.
7. Ask only the minimum necessary clarifying question.
8. Do not invent tool names or arguments.
9. If the user asks for something outside policy or impossible with available tools, use respond.
10. If transfer is required by policy and transfer_to_human_agents is available, call it first.
"""


def _normalize_tools(tools: list) -> list:
    normalized = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue

        # Already Anthropic-style tool
        if t.get("name") and t.get("input_schema"):
            normalized.append(t)
            continue

        # OpenAI-style function tool
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            name = fn.get("name")
            description = fn.get("description", "")
            parameters = fn.get("parameters", {"type": "object", "properties": {}})
            if name:
                normalized.append(
                    {
                        "name": name,
                        "description": description,
                        "input_schema": parameters,
                    }
                )
            continue

        # Flat tool dict
        if t.get("name") and t.get("parameters"):
            normalized.append(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t["parameters"],
                }
            )

    return normalized


def _extract_text_from_response(response) -> str:
    text_parts = []
    for block in response.content:
        if hasattr(block, "text") and block.text:
            text_parts.append(block.text)
    return "".join(text_parts).strip()


def _safe_json_action(text_response: str) -> str:
    if not text_response:
        return json.dumps({"name": "respond", "arguments": {"content": ""}})

    try:
        parsed = json.loads(text_response)
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            return json.dumps(parsed)
    except Exception:
        pass

    return json.dumps({"name": "respond", "arguments": {"content": text_response}})


def run_agent(task: str, tools: list, conversation_history: list) -> tuple:
    print("PRINT_MARKER_RUN_AGENT_ENTERED_DEBUG_V6", flush=True)

    client = get_client()
    model = get_model()
    messages = conversation_history.copy()

    logger.info(f"run_agent called with task: {task[:300] if task else ''}")
    logger.info(f"run_agent tools count raw: {len(tools) if tools else 0}")
    logger.info(f"run_agent history length: {len(messages)}")

    print(f"PRINT_MARKER_RUN_AGENT_TOOLS_COUNT_RAW={len(tools) if tools else 0}", flush=True)
    print(f"PRINT_MARKER_RUN_AGENT_HISTORY_LEN={len(messages)}", flush=True)

    if not messages:
        messages = [{"role": "user", "content": task}]
    elif task and messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": task})

    anthropic_tools = _normalize_tools(tools)
    logger.info(f"run_agent tools count normalized: {len(anthropic_tools)}")
    print(f"PRINT_MARKER_RUN_AGENT_TOOLS_COUNT_NORMALIZED={len(anthropic_tools)}", flush=True)

    kwargs = {
        "model": model,
        "max_tokens": 1200,
        "temperature": 0,
        "system": SYSTEM_PREFIX,
        "messages": messages,
    }

    if anthropic_tools:
        kwargs["tools"] = anthropic_tools
        logger.info(f"Passing tools to Anthropic: {[t['name'] for t in anthropic_tools]}")
        print(
            f"PRINT_MARKER_TOOLS_AVAILABLE names={[t['name'] for t in anthropic_tools]}",
            flush=True,
        )
    else:
        logger.info("No tools available after normalization")
        print("PRINT_MARKER_NO_TOOLS_AVAILABLE", flush=True)

    response = client.messages.create(**kwargs)

    logger.info(
        f"Single call response: stop_reason={response.stop_reason}, blocks={len(response.content)}"
    )
    print(
        f"PRINT_MARKER_AGENT_RESPONSE stop_reason={response.stop_reason} blocks={len(response.content)}",
        flush=True,
    )

    # If Claude actually emits a tool_use block, convert it directly to the expected action JSON.
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            action = {
                "name": getattr(block, "name", "respond"),
                "arguments": getattr(block, "input", {}) or {},
            }
            logger.info(f"Returning tool_use action: {action}")
            print("PRINT_MARKER_AGENT_RETURN_TOOL_USE_V6", flush=True)
            action_json = json.dumps(action)
            messages.append({"role": "assistant", "content": action_json})
            return action_json, messages

    text_response = _extract_text_from_response(response)
    action_json = _safe_json_action(text_response)

    logger.info(f"Returning final action json length={len(action_json)}")
    print("PRINT_MARKER_AGENT_RETURN_FINAL_JSON_V6", flush=True)

    messages.append({"role": "assistant", "content": action_json})
    return action_json, messages
