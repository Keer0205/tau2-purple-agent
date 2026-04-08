import json
import logging
from typing import Any
from typing_extensions import override

print("PRINT_MARKER_EXECUTOR_MODULE_LOADED_DEBUG_V8", flush=True)

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


def _extract_tools_from_context(context: RequestContext) -> list:
    candidate_paths = [
        ("context.tools", _safe_getattr(context, "tools")),
        ("context.available_tools", _safe_getattr(context, "available_tools")),
        ("context.message.tools", _safe_getattr(_safe_getattr(context, "message"), "tools")),
        ("context.message.metadata", _safe_getattr(_safe_getattr(context, "message"), "metadata")),
        ("context.metadata", _safe_getattr(context, "metadata")),
    ]

    for label, value in candidate_paths:
        logger.info(f"DEBUG tool source check: {label} -> {type(value)} :: {value}")

        if isinstance(value, list):
            logger.info(f"DEBUG using tools from {label}, count={len(value)}")
            return value

        if isinstance(value, dict):
            for key in ["tools", "available_tools", "functions", "actions"]:
                if isinstance(value.get(key), list):
                    logger.info(
                        f"DEBUG using tools from {label}['{key}'], count={len(value.get(key))}"
                    )
                    return value.get(key)

    logger.warning("DEBUG no tools found in known context locations; defaulting to []")
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

    import re

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
        print("PRINT_MARKER_EXECUTOR_INIT_DEBUG_V8", flush=True)
        super().__init__()

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("PRINT_MARKER_EXECUTOR_EXECUTE_ENTERED_DEBUG_V8", flush=True)

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Thinking..."),
        )

        input_text = get_message_text(context.message)
        logger.info(f"Task {context.task_id}: received: {input_text[:500]}")

        task_id = context.task_id
        if task_id not in _task_histories:
            _task_histories[task_id] = []

        history = _task_histories[task_id]

        try:
            logger.info(f"DEBUG context type: {type(context)}")
            logger.info(f"DEBUG context attrs: {dir(context)}")
            logger.info(f"DEBUG message object: {context.message}")

            tools = _extract_tools_from_context(context)
            logger.info(f"DEBUG final tools count: {len(tools)}")
            logger.info(f"DEBUG final tools value: {tools}")

            print(
                f"PRINT_MARKER_EXECUTOR_V4_TOOLS count={len(tools)} tools={tools}",
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
