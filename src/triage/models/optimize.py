"""Otimização de latência: exporta o classificador para ONNX Runtime.

Técnica escolhida (dentre as vistas na Fase 3): **conversão para ONNX**.

Decisão de escopo: convertemos apenas o estágio de classificação
(LogisticRegression) para ONNX via `skl2onnx`. O estágio de vetorização
(TfidfVectorizer) permanece em scikit-learn — a conversão nativa de texto
para ONNX exige operadores extras (`onnxruntime-extensions`) e, para um
vocabulário de ~20k termos, o `transform` do TF-IDF já é uma fração pequena
da latência total (é a matriz esparsa vezes os pesos do classificador que
domina o custo em lote). Isso é documentado aqui e no README/Model Card
como uma limitação consciente, não uma omissão.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from itertools import cycle

import joblib
import numpy as np
import onnxruntime as rt
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from triage.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Estatísticas de latência (em milissegundos) de N inferências."""

    n_requests: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def export_classifier_to_onnx(settings: Settings) -> None:
    """Carrega o pipeline treinado e exporta vetorizador + classificador ONNX."""
    pipeline = joblib.load(settings.baseline_model_path)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    n_features = len(vectorizer.vocabulary_)
    onnx_model = to_onnx(
        classifier,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=17,
    )

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.onnx_classifier_path.write_bytes(onnx_model.SerializeToString())
    joblib.dump(vectorizer, settings.onnx_vectorizer_path)
    logger.info("Classificador ONNX salvo em %s", settings.onnx_classifier_path)


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct))


def _time_calls(fn, n_requests: int, warmup: int = 20) -> LatencyStats:
    """Cronometra `n_requests` chamadas de `fn`, descartando `warmup` chamadas
    iniciais (aquecimento de cache/JIT, para reduzir ruído nas medições)."""
    for _ in range(warmup):
        fn()

    durations_ms: list[float] = []
    for _ in range(n_requests):
        start = time.perf_counter()
        fn()
        durations_ms.append((time.perf_counter() - start) * 1000)
    return LatencyStats(
        n_requests=n_requests,
        mean_ms=float(np.mean(durations_ms)),
        p50_ms=_percentile(durations_ms, 50),
        p95_ms=_percentile(durations_ms, 95),
        p99_ms=_percentile(durations_ms, 99),
    )


def benchmark_baseline(
    settings: Settings, texts: list[str], n_requests: int
) -> LatencyStats:
    """Latência do pipeline scikit-learn original (TF-IDF + LR), 1 texto por vez."""
    pipeline = joblib.load(settings.baseline_model_path)
    sample = cycle(texts)
    return _time_calls(lambda: pipeline.predict([next(sample)]), n_requests)


def benchmark_onnx(
    settings: Settings, texts: list[str], n_requests: int
) -> LatencyStats:
    """Latência do classificador via ONNX Runtime (vetorização sklearn + ONNX)."""
    vectorizer = joblib.load(settings.onnx_vectorizer_path)
    session = rt.InferenceSession(
        str(settings.onnx_classifier_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    sample = cycle(texts)

    def _predict_one() -> None:
        vec = vectorizer.transform([next(sample)]).toarray().astype(np.float32)
        session.run(None, {input_name: vec})

    return _time_calls(_predict_one, n_requests)


def run_latency_comparison(settings: Settings, n_requests: int = 1000) -> dict:
    """Roda o comparativo baseline vs. ONNX e salva em reports/."""
    from triage.data.loader import load_dataset

    _, test_df = load_dataset(settings.raw_train_path, settings.raw_test_path)
    texts = test_df["medical_abstract"].tolist()

    baseline_stats = benchmark_baseline(settings, texts, n_requests)
    onnx_stats = benchmark_onnx(settings, texts, n_requests)

    speedup_p50 = (
        baseline_stats.p50_ms / onnx_stats.p50_ms if onnx_stats.p50_ms else float("nan")
    )
    speedup_p95 = (
        baseline_stats.p95_ms / onnx_stats.p95_ms if onnx_stats.p95_ms else float("nan")
    )

    result = {
        "n_requests": n_requests,
        "baseline_sklearn": asdict(baseline_stats),
        "optimized_onnxruntime": asdict(onnx_stats),
        "speedup_p50_x": round(speedup_p50, 2),
        "speedup_p95_x": round(speedup_p95, 2),
    }

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.reports_dir / "latency_comparison.json"
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Comparativo de latência salvo em %s", output_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    export_classifier_to_onnx(settings)
    comparison = run_latency_comparison(settings)
    print(json.dumps(comparison, indent=2))
