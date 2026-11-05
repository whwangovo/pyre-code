FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl graphviz \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (~200MB vs ~2GB with CUDA)
RUN pip install --no-cache-dir "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu

COPY grading_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY torch_judge/ /app/torch_judge/
COPY grading_service/ /app/grading_service/
COPY pyproject.toml /app/pyproject.toml

RUN mkdir -p /app/data \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /app
USER app

ENV DB_PATH=/app/data/pyre.db

EXPOSE 8000

CMD ["uvicorn", "grading_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
