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

            user_input = ""
            tools = []

            if context.message and context.message.parts:
                for part in context.message.parts:
                    # Extract text
                    if hasattr(part, 'root'):
                        root = part.root
                        if hasattr(root, 'text') and root.text:
                            user_input += root.text
                        # Extract tools from data part
                        elif hasattr(root, 'data') and root.data:
                            try:
                                data = root.data
                                if isinstance(data, dict) and 'tools' in data:
                                    tools = data['tools']
                                elif isinstance(data, list):
                                    tools = data
                            except Exception as e:
                                logger.warning(f"Could not parse tools from data: {e}")
                    elif hasattr(part, 'text') and part.text:
                        user_input += part.text

            logger.info(f"Task {task_id}: input: {user_input[:100]}, tools: {len(tools)}")

            history = self.conversation_history.get(task_id, [])
            response_text, updated_history = run_agent(user_input, tools, history)
            self.conversation_history[task_id] = updated_history

            await event_queue.enqueue_event(new_agent_text_message(response_text))

        except Exception as e:
            logger.error(f"Executor error: {e}", exc_info=True)
            await event_queue.enqueue_event(
                new_agent_text_message(f"I apologize, I encountered an error: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
