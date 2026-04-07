import os
import logging
import anthropic
from typing_extensions import override
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import get_message_text, new_agent_text_message

logger = logging.getLogger(__name__)


def get_client():
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY") or
        os.environ.get("AMBER_CONFIG_AGENT_ANTHROPIC_API_KEY")
    )
    return anthropic.Anthropic(api_key=api_key)


SYSTEM_PROMPT = """You are an expert customer service agent for airline, retail, and telecom domains.

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


# Store conversation history per task_id
_task_histories: dict[str, list] = {}


class Executor(AgentExecutor):
    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # Signal we are working
        await updater.update_status(TaskState.working, new_agent_text_message("Processing..."))

        # Get the incoming user message text
        user_text = get_message_text(context.message)
        logger.info(f"Task {context.task_id}: received message: {user_text[:100]}")

        # Maintain conversation history per task
        task_id = context.task_id
        if task_id not in _task_histories:
            _task_histories[task_id] = []

        history = _task_histories[task_id]
        history.append({"role": "user", "content": user_text})

        try:
            client = get_client()
            agent_llm = os.environ.get("AGENT_LLM", "anthropic/claude-3-5-sonnet-20241022")
            model = agent_llm.split("/")[-1] if "/" in agent_llm else agent_llm

            max_iterations = 10
            response_text = ""

            for iteration in range(max_iterations):
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=history,
                )
                logger.info(f"Iteration {iteration}: stop_reason={response.stop_reason}")

                text_parts = []
                tool_use_blocks = []

                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_use_blocks.append(block)

                response_text = "".join(text_parts)
                history.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn" or not tool_use_blocks:
                    break

                # Handle tool calls (tau2 tools arrive as tool_use blocks)
                tool_results = []
                for tool_block in tool_use_blocks:
                    logger.info(f"Tool call: {tool_block.name}({tool_block.input})")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": f"Tool '{tool_block.name}' is not available in this environment.",
                    })

                history.append({"role": "user", "content": tool_results})

            # Send final response
            await updater.add_artifact(
                parts=[{"kind": "text", "text": response_text}],
                name="response",
            )
            await updater.update_status(TaskState.completed, new_agent_text_message(response_text), final=True)

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {str(e)}"),
                final=True,
            )

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")
