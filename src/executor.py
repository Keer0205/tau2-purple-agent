import logging
import json
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from agent import run_agent

logger = logging.getLogger(__name__)

class Executor(AgentExecutor):
    def __init__(self):
        self.conversation_history = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            task_id = context.task_id

            # Get user message from A2A context
            user_input = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        user_input += part.root.text
                    elif hasattr(part, 'text'):
                        user_input += part.text

            logger.info(f"Task {task_id}: received input: {user_input[:100]}")

            # Extract tools from context
            tools = []
            if hasattr(context, 'tools') and context.tools:
                tools = context.tools
                logger.info(f"Task {task_id}: found {len(tools)} tools")

            # Get conversation history
            history = self.conversation_history.get(task_id, [])

            # Run agent with tools
            response_text, updated_history = run_agent(user_input, tools, history)
            self.conversation_history[task_id] = updated_history

            logger.info(f"Task {task_id}: response: {response_text[:100]}")

            # Send response
            await event_queue.enqueue_event(new_agent_text_message(response_text))

        except Exception as e:
            logger.error(f"Executor error: {e}", exc_info=True)
            await event_queue.enqueue_event(
                new_agent_text_message(f"I apologize, I encountered an error: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
