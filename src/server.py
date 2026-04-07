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
