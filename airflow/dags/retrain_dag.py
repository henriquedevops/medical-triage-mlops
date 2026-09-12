"""DAG de retreino do classificador de triagem.

Orquestra o pipeline mínimo pedido pelo enunciado: ingestão de dados ->
treino -> salvamento do modelo. A lógica de cada etapa vive em
`triage.data.loader` e `triage.models.train` (testada por `pytest`
independentemente do Airflow); este arquivo só faz a *orquestração*.

Para rodar localmente:
    export AIRFLOW_HOME=~/airflow
    airflow dags test medical_triage_retrain 2026-01-01
"""

from __future__ import annotations

import logging
from datetime import datetime

from airflow.decorators import dag, task

logger = logging.getLogger(__name__)


@dag(
    dag_id="medical_triage_retrain",
    description="Pipeline de retreino do classificador de urgência de laudos",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["tech-challenge-fase3", "mlops", "triage"],
)
def medical_triage_retrain():
    """Ingestão -> treino -> salvamento do modelo de triagem."""

    @task()
    def load_data() -> dict:
        """Carrega e valida os splits de treino/teste a partir do CSV bruto."""
        from triage.config import Settings
        from triage.data.loader import load_dataset

        settings = Settings()
        train_df, test_df = load_dataset(
            settings.raw_train_path, settings.raw_test_path
        )
        logger.info("train=%d linhas, test=%d linhas", len(train_df), len(test_df))
        return {"n_train": len(train_df), "n_test": len(test_df)}

    @task()
    def train_and_save_model(ingestion_stats: dict) -> dict:
        """Treina o pipeline TF-IDF + LR e persiste o artefato em disco."""
        from triage.config import Settings
        from triage.models.train import run_training_job

        logger.info("Iniciando treino sobre %s", ingestion_stats)
        result = run_training_job(Settings())
        return result.metrics

    @task()
    def report_metrics(metrics: dict) -> None:
        """Loga as métricas finais (em produção: publicaria em um dashboard)."""
        logger.info("Métricas do retreino: %s", metrics)

    ingestion_stats = load_data()
    metrics = train_and_save_model(ingestion_stats)
    report_metrics(metrics)


medical_triage_retrain()
