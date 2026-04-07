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

print("PRINT_MARKER_SERVER_MODULE_LOADED_DEBUG_V9", flush=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("PRINT_MARKER_SERVER_MAIN_START_DEBUG_V9", flush=True)
    print(f"PRINT_MARKER_AGENT_VERSION={os.getenv('AGENT_VERSION', 'unset')}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card_url", default=None)
    args = parser.parse_args()

    skill = AgentSkill(
        id="airline_customer_service",
        name="Airline Customer Service",
        description=(
            "Handles airline customer service tasks such as flight booking help, "
            "flight changes, cancellations, refunds, baggage questions, itinerary support, "
            "passenger request handling, and airline policy guidance."
        ),
        tags=[
            "airline",
            "customer-service",
            "travel",
            "flight",
            "booking",
            "reservation",
            "refund",
            "baggage",
            "itinerary",
        ],
        examples=[
            "I need to cancel my flight",
            "Help me change my booking",
            "What is the baggage allowance for my trip?",
            "I want a refund for my airline ticket",
            "Can you help with my flight itinerary?",
        ],
    )

    agent_card = AgentCard(
        name="Purple Airline Customer Service Agent",
        description=(
            "An A2A airline customer service agent for tau2-bench evaluation. "
            "Specialized in airline support tasks including booking help, cancellations, "
            "refunds, baggage queries, and itinerary assistance."
        ),
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
