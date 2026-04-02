FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
ARG ANTHROPIC_API_KEY
ENV ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
RUN pip install --upgrade pip && \
    pip install "a2a-sdk[http-server]>=0.2.0" anthropic uvicorn
COPY src/ ./src/
WORKDIR /app/src
EXPOSE 9009
ENTRYPOINT ["python", "-u", "server.py", "--host", "0.0.0.0", "--port", "9009"]
