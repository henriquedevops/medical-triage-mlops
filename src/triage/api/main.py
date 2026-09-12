"""API REST de triagem de laudos médicos (FastAPI + instrumentação Prometheus).

Endpoints:
- `POST /triage`   — classifica um laudo em normal / atencao / urgente.
- `GET  /health`   — liveness/readiness simples.
- `GET  /metrics`  — métricas no formato Prometheus (contagem, latência, erros).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from triage.config import Settings
from triage.models.predict import TriagePredictor

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "triage_requests_total",
    "Total de requisições recebidas pela API de triagem",
    ["endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "triage_request_latency_seconds",
    "Latência das requisições da API de triagem",
    ["endpoint"],
)
PREDICTION_COUNT = Counter(
    "triage_predictions_total",
    "Total de predições por nível de urgência",
    ["urgency"],
)
ERROR_COUNT = Counter(
    "triage_errors_total",
    "Total de erros na API de triagem",
    ["endpoint"],
)

_predictor: TriagePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    settings = Settings()
    _predictor = TriagePredictor(settings)
    logger.info("Modelo carregado (backend=%s)", _predictor.backend)
    yield


app = FastAPI(title=Settings().api_title, lifespan=lifespan)


class TriageRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Texto do laudo/exame a ser classificado.",
        examples=["Paciente com dor torácica súbita e sudorese, suspeita de IAM."],
    )


class TriageResponse(BaseModel):
    urgency: str
    scores: dict[str, float]
    model_backend: str


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Instrumenta toda requisição HTTP com contagem e latência Prometheus."""
    endpoint = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        REQUEST_COUNT.labels(endpoint=endpoint, status_code="500").inc()
        raise
    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    REQUEST_COUNT.labels(endpoint=endpoint, status_code=str(response.status_code)).inc()
    if response.status_code >= 500:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
    return response


@app.get("/health")
def health() -> dict:
    backend = _predictor.backend if _predictor else "unloaded"
    return {"status": "ok", "model_backend": backend}


@app.post("/triage", response_model=TriageResponse)
def triage(payload: TriageRequest) -> TriageResponse:
    assert _predictor is not None, "Modelo não carregado — verifique o lifespan da API."
    result = _predictor.predict(payload.text)
    PREDICTION_COUNT.labels(urgency=result["urgency"]).inc()
    return TriageResponse(
        urgency=result["urgency"],
        scores=result["scores"],
        model_backend=_predictor.backend,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
