FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install anthropic a2a-sdk uvicorn
COPY src/ ./src/
WORKDIR /app/src
CMD ["python", "server.py"]
