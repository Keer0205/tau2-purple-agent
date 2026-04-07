import argparse
import logging
import os
import socket

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from executor import Executor

print("PRINT_MARKER_SERVER_MODULE_LOADED_DEBUG_V6", flush=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("PRINT_MARKER_SERVER_MAIN_START_DEBUG_V6", flush=True)
    print(f"PRINT_MARKER_AGENT_VERSION={os.getenv('AGENT_VERSION', 'unset')}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card_url", default=None)
    args = parser.parse_args()

    skill = AgentSkill(
        id="tau2-customer-service",
        name="Customer Service Agent",
        description=(
            "Handles customer service tasks for airline, retail, and telecom domains. "
            "Can look up accounts, process refunds, change plans, troubleshoot issues, "
            "and coordinate with users to fully resolve their requests."
        ),
        tags=["customer-service", "airline", "retail", "telecom"],
        examples=[
            "I need to cancel my flight",
            "Where is my order?",
            "Change my phone plan",
            "My mobile data isn't working",
            "I want a refund for my purchase",
        ],
    )

    agent_card = AgentCard(
        name="Tau2 Purple Agent",
        description="A Claude-powered customer service agent for tau2-bench evaluation",
        url=args.card_url or f"http://{socket.gethostname()}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    executor = Executor()
    print(f"PRINT_MARKER_SERVER_EXECUTOR_INSTANCE={type(executor)}", flush=True)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
