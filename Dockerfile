FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install anthropic uvicorn && \
    pip install a2a-sdk || pip install "a2a[server]" || pip install a2a
COPY src/ ./src/
WORKDIR /app/src
EXPOSE 9009
ENTRYPOINT ["python", "server.py", "--host", "0.0.0.0", "--port", "9009"]
