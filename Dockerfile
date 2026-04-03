FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install "a2a-sdk[http-server]>=0.2.0" anthropic uvicorn
COPY src/ ./src/
EXPOSE 9009
CMD ["python", "-u", "src/server.py", "--host", "0.0.0.0", "--port", "9009"]
