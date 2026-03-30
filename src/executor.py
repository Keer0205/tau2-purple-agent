import json
import logging
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
            user_input = context.get_user_input()
            tools = context.get_tools() or []
            
            history = self.conversation_history.get(task_id, [])
            if user_input:
                history.append({"role": "user", "content": user_input})
            
            response_text, updated_history = run_agent(user_input, tools, history)
            self.conversation_history[task_id] = updated_history
            
            await event_queue.enqueue_event(new_agent_text_message(response_text))
        except Exception as e:
            logger.error(f"Executor error: {e}")
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
