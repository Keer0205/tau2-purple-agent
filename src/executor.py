import json
import logging
import re
from typing import Any
from typing_extensions import override

print("PRINT_MARKER_EXECUTOR_MODULE_LOADED_DEBUG_V9", flush=True)

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message

from agent import run_agent

logger = logging.getLogger(__name__)

_task_histories: dict[str, list] = {}


def _safe_getattr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _extract_tool_list_from_text(task_text: str) -> list:
    """
    Fallback extractor for benchmarks that embed tool definitions directly
    inside the prompt text rather than passing them via structured context.
    """
    if not task_text:
        return []

    marker = "Here's a list of tools you can use"
    start_idx = task_text.find(marker)
    if start_idx == -1:
        return []

    # Find first '[' after the marker
    list_start = task_text.find("[", start_idx)
    if list_start == -1:
        return []

    # Find the matching closing ']'
    depth = 0
    list_end = -1
    for i in range(list_start, len(task_text)):
        ch = task_text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                list_end = i
                break

    if list_end == -1:
        logger.warning("DEBUG fallback tool parser could not find closing bracket")
        return []

    raw_tools_block = task_text[list_start:list_end + 1]

    try:
        parsed = json.loads(raw_tools_block)
        if isinstance(parsed, list):
            logger.info(
                f"DEBUG fallback parsed tools from task text, count={len(parsed)}"
            )
            return parsed
    except Exception as e:
        logger.warning(f"DEBUG fallback tool parser failed: {e}")

    return []


def _extract_tools_from_context(context: RequestContext, task_text: str) -> list:
    candidate_paths = [
        ("context.tools", _safe_getattr(context, "tools")),
        ("context.available_tools", _safe_getattr(context, "available_tools")),
        ("context.message.tools", _safe_getattr(_safe_getattr(context, "message"), "tools")),
        ("context.message.metadata", _safe_getattr(_safe_getattr(context, "message"), "metadata")),
        ("context.metadata", _safe_getattr(context, "metadata")),
    ]

    for label, value in candidate_paths:
        logger.info(f"DEBUG tool source check: {label} -> {type(value)} :: {value}")

        if isinstance(value, list) and value:
            logger.info(f"DEBUG using tools from {label}, count={len(value)}")
            return value

        if isinstance(value, dict):
            for key in ["tools", "available_tools", "functions", "actions"]:
                candidate = value.get(key)
                if isinstance(candidate, list) and candidate:
                    logger.info(
                        f"DEBUG using tools from {label}['{key}'], count={len(candidate)}"
                    )
                    return candidate

    logger.warning("DEBUG no structured tools found; trying fallback parse from task text")
    parsed_tools = _extract_tool_list_from_text(task_text)
    if parsed_tools:
        return parsed_tools

    logger.warning("DEBUG no tools found anywhere; defaulting to []")
    return []


def _to_action_json(text_response: str) -> dict:
    if not text_response or not text_response.strip():
        return {"name": "respond", "arguments": {"content": ""}}

    try:
        parsed = json.loads(text_response)
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text_response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                return parsed
        except Exception:
            pass

    return {"name": "respond", "arguments": {"content": text_response}}


class Executor(AgentExecutor):
    def __init__(self):
        print("PRINT_MARKER_EXECUTOR_INIT_DEBUG_V9", flush=True)
        super().__init__()

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("PRINT_MARKER_EXECUTOR_EXECUTE_ENTERED_DEBUG_V9", flush=True)

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Thinking..."),
        )

        input_text = get_message_text(context.message)
        logger.info(f"Task {context.task_id}: received: {input_text[:1000]}")

        task_id = context.task_id
        if task_id not in _task_histories:
            _task_histories[task_id] = []

        history = _task_histories[task_id]

        try:
            logger.info(f"DEBUG context type: {type(context)}")
            logger.info(f"DEBUG context attrs: {dir(context)}")
            logger.info(f"DEBUG message object: {context.message}")

            tools = _extract_tools_from_context(context, input_text)
            logger.info(f"DEBUG final tools count: {len(tools)}")
            logger.info(f"DEBUG final tools value: {tools}")

            print(
                f"PRINT_MARKER_EXECUTOR_V9_TOOLS count={len(tools)}",
                flush=True,
            )

            text_response, updated_history = run_agent(
                task=input_text,
                tools=tools,
                conversation_history=history,
            )

            _task_histories[task_id] = updated_history

            assistant_json = _to_action_json(text_response)
            logger.info(f"DEBUG final assistant_json: {assistant_json}")

            await updater.add_artifact(
                parts=[Part(root=DataPart(data=assistant_json))],
                name="Action",
            )

            await updater.update_status(
                TaskState.completed,
                final=True,
            )

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)

            fallback = {
                "name": "respond",
                "arguments": {"content": f"Error: {str(e)}"},
            }

            await updater.add_artifact(
                parts=[Part(root=DataPart(data=fallback))],
                name="Action",
            )

            await updater.update_status(
                TaskState.failed,
                final=True,
            )

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")
