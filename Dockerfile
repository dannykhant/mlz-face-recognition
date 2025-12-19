FROM python:3.13-slim-bookworm

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

COPY face_recognition_v202510191652.onnx labels.json ./
COPY predict.py ./

ENV PATH="/app/.venv/bin:$PATH"

CMD ["fastapi", "run", "predict.py"]