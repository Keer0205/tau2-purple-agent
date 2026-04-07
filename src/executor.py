import json
import os
import logging
from typing_extensions import override

import anthropic
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful customer service agent. "
    "Follow the policy and tool instructions provided in each message. "
    "Always respond with a JSON object with 'name' and 'arguments' fields. "
    "To respond to the user: {\"name\": \"respond\", \"arguments\": {\"content\": \"your message\"}}. "
    "To call a tool: {\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}}. "
    "Only output valid JSON, nothing else."
)

# Store conversation history per task_id
_task_histories: dict[str, list] = {}


class Executor(AgentExecutor):
    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.update_status(TaskState.working, new_agent_text_message("Thinking..."))

        input_text = get_message_text(context.message)
        logger.info(f"Task {context.task_id}: received: {input_text[:200]}")

        # Maintain conversation history per task
        task_id = context.task_id
        if task_id not in _task_histories:
            _task_histories[task_id] = [{"role": "user", "content": input_text}]
        else:
            _task_histories[task_id].append({"role": "user", "content": input_text})

        history = _task_histories[task_id]

        try:
            agent_llm = os.environ.get("AGENT_LLM", "anthropic/claude-3-5-sonnet-20241022")
            model = agent_llm.split("/")[-1] if "/" in agent_llm else agent_llm

            client = anthropic.Anthropic(
                api_key=(
                    os.environ.get("ANTHROPIC_API_KEY") or
                    os.environ.get("AMBER_CONFIG_AGENT_ANTHROPIC_API_KEY")
                )
            )

            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=history,
            )

            assistant_content = response.content[0].text if response.content else "{}"

            # Parse JSON response
            try:
                assistant_json = json.loads(assistant_content)
            except json.JSONDecodeError:
                # Extract JSON if wrapped in markdown
                import re
                match = re.search(r'\{.*\}', assistant_content, re.DOTALL)
                if match:
                    assistant_json = json.loads(match.group())
                else:
                    assistant_json = {"name": "respond", "arguments": {"content": assistant_content}}

            history.append({"role": "assistant", "content": assistant_content})
            logger.info(f"Response JSON: {assistant_json}")

            await updater.add_artifact(
                parts=[Part(root=DataPart(data=assistant_json))],
                name="Action",
            )
            await updater.update_status(
                TaskState.completed,
                new_agent_text_message(json.dumps(assistant_json)),
                final=True,
            )

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            fallback = {"name": "respond", "arguments": {"content": f"Error: {str(e)}"}}
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=fallback))],
                name="Action",
            )
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(str(e)),
                final=True,
            )

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")
