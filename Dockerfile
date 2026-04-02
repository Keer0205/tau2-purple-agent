FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install .
COPY src/ ./src/
WORKDIR /app/src
EXPOSE 9009
ENTRYPOINT ["python", "-u", "server.py", "--host", "0.0.0.0", "--port", "9009"]
