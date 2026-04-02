FROM python:3.11-slim
WORKDIR /app
RUN pip install --upgrade pip && \
    pip install anthropic>=0.40.0 "a2a-sdk>=0.2.0" uvicorn>=0.30.0
COPY src/ ./src/
WORKDIR /app/src
EXPOSE 9009
ENTRYPOINT ["python", "-u", "server.py", "--host", "0.0.0.0", "--port", "9009"]
