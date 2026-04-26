FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md openenv.yaml ./
COPY supportops_x_env ./supportops_x_env
COPY training ./training
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000

CMD ["uvicorn", "supportops_x_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
