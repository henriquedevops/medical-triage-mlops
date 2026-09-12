# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv==0.11.28

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md ./

# --frozen: instala exatamente o que está no lock (build reprodutível).
# --no-dev: não traz ruff/pytest/httpx para a imagem final.
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY README.md ./
COPY data/raw ./data/raw

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    BASELINE_MODEL_PATH=/app/models/triage_pipeline.joblib

# O modelo é treinado no build para a imagem já subir pronta para servir
# (dataset pequeno e commitado no repo -> treino determinístico em ~6s,
# ver src/triage/config.py e README, seção "Execução").
RUN mkdir -p /app/models /app/reports && \
    python -m triage.models.train

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "triage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
