import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from agent import run_agent

logger = logging.getLogger(__name__)


def extract_tools_from_context(context: RequestContext) -> list:
    """
    Extract MCP/A2A tools from the request context so the agent can use them.
    Returns a list of tool dicts in Anthropic format.
    """
    tools = []
    try:
        # A2A passes tools via context metadata or capabilities
        if hasattr(context, 'tools') and context.tools:
            for tool in context.tools:
                tools.append({
                    "name": tool.name,
                    "description": getattr(tool, 'description', ''),
                    "input_schema": getattr(tool, 'input_schema', {"type": "object", "properties": {}}),
                })
        # Some A2A versions nest tools differently
        elif hasattr(context, 'metadata') and context.metadata:
            raw_tools = context.metadata.get('tools', [])
            for t in raw_tools:
                tools.append({
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema", {"type": "object", "properties": {}}),
                })
    except Exception as e:
        logger.warning(f"Could not extract tools from context: {e}")
    
    logger.info(f"Extracted {len(tools)} tools from context")
    return tools


def extract_user_input(context: RequestContext) -> str:
    """Safely extract text from all message parts."""
    user_input = ""
    try:
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    user_input += part.root.text
                elif hasattr(part, 'text'):
                    user_input += part.text
    except Exception as e:
        logger.warning(f"Could not extract user input: {e}")
    return user_input.strip()


class Executor(AgentExecutor):
    def __init__(self):
        # Per-task conversation history
        self.conversation_history: dict[str, list] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            task_id = context.task_id
            user_input = extract_user_input(context)
            tools = extract_tools_from_context(context)

            logger.info(f"Task {task_id}: input='{user_input[:120]}' tools={len(tools)}")

            if not user_input:
                logger.warning(f"Task {task_id}: empty input received")
                await event_queue.enqueue_event(
                    new_agent_text_message("I didn't receive any input. Please describe the task.")
                )
                return

            history = self.conversation_history.get(task_id, [])
            response_text, updated_history = run_agent(user_input, tools, history)
            self.conversation_history[task_id] = updated_history

            logger.info(f"Task {task_id}: response length={len(response_text)}")
            await event_queue.enqueue_event(new_agent_text_message(response_text))

        except Exception as e:
            logger.error(f"Executor error on task {context.task_id}: {e}", exc_info=True)
            await event_queue.enqueue_event(
                new_agent_text_message(f"I encountered an error processing your request: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        if task_id in self.conversation_history:
            del self.conversation_history[task_id]
        logger.info(f"Task {task_id} cancelled")
