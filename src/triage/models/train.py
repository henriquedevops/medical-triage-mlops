"""Treino do classificador de urgência (TF-IDF + Regressão Logística).

Modelo leve por decisão de projeto (ver README, seção "Decisão Arquitetural"):
o tema da fase é o ciclo de vida em produção (CI/CD, orquestração, latência,
monitoramento), não a sofisticação do modelo. TF-IDF + LogisticRegression é
o baseline sugerido pelo próprio enunciado ("TF-IDF + Random Forest ou
modelo leve similar") e treina em segundos, o que também facilita o DAG de
retreino do Airflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from triage.config import Settings, seed_everything
from triage.data.loader import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Resultado de uma rodada de treino: pipeline ajustado + métricas de teste."""

    pipeline: Pipeline
    metrics: dict[str, float]
    report: str


def build_pipeline(settings: Settings) -> Pipeline:
    """Monta o pipeline scikit-learn (vetorização + classificador)."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=settings.tfidf_max_features,
                    ngram_range=(1, settings.tfidf_ngram_max),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=settings.seed,
                ),
            ),
        ]
    )


def train_and_evaluate(
    settings: Settings,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> TrainResult:
    """Treina o pipeline e avalia no split de teste.

    Aceita `train_df`/`test_df` já carregados (usado pelo DAG do Airflow, que
    separa a etapa de ingestão da etapa de treino); se omitidos, carrega do
    disco a partir das configurações.
    """
    seed_everything(settings.seed)

    if train_df is None or test_df is None:
        train_df, test_df = load_dataset(
            settings.raw_train_path, settings.raw_test_path
        )

    pipeline = build_pipeline(settings)
    pipeline.fit(train_df["medical_abstract"], train_df["urgency"])

    predictions = pipeline.predict(test_df["medical_abstract"])
    macro_f1 = f1_score(test_df["urgency"], predictions, average="macro")
    weighted_f1 = f1_score(test_df["urgency"], predictions, average="weighted")
    report = classification_report(test_df["urgency"], predictions)

    logger.info("macro_f1=%.4f weighted_f1=%.4f", macro_f1, weighted_f1)

    return TrainResult(
        pipeline=pipeline,
        metrics={"macro_f1": macro_f1, "weighted_f1": weighted_f1},
        report=report,
    )


def save_pipeline(pipeline: Pipeline, output_path: Path) -> None:
    """Persiste o pipeline treinado (vetorizador + classificador) em disco."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Modelo salvo em %s", output_path)


def run_training_job(settings: Settings | None = None) -> TrainResult:
    """Ponto de entrada único usado pela CLI, pelos testes e pela DAG do Airflow."""
    settings = settings or Settings()
    result = train_and_evaluate(settings)
    save_pipeline(result.pipeline, settings.baseline_model_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_training_job()
    print(result.report)
