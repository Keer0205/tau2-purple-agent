FROM python:3.11-slim

WORKDIR /app/src

COPY pyproject.toml /app/
RUN pip install --no-cache-dir /app

COPY src/ /app/src/

ENV PYTHONUNBUFFERED=1
ENV AGENT_VERSION=debug-v7

EXPOSE 9009

ENTRYPOINT ["python", "server.py"]
