"""Configurações centralizadas do projeto via Pydantic Settings.

Todas as constantes que variam entre ambientes (paths, hiperparâmetros de
runtime, seed) vivem aqui. Nada de valores mágicos espalhados pelo código.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configurações da aplicação, sobrescrevíveis via variáveis de ambiente."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="TRIAGE_", extra="ignore"
    )

    seed: int = 42

    raw_train_path: Path = BASE_DIR / "data" / "raw" / "medical_tc_train.csv"
    raw_test_path: Path = BASE_DIR / "data" / "raw" / "medical_tc_test.csv"
    raw_labels_path: Path = BASE_DIR / "data" / "raw" / "medical_tc_labels.csv"
    processed_dir: Path = BASE_DIR / "data" / "processed"

    model_dir: Path = BASE_DIR / "models"
    baseline_model_path: Path = BASE_DIR / "models" / "triage_pipeline.joblib"
    onnx_classifier_path: Path = BASE_DIR / "models" / "triage_classifier.onnx"
    onnx_vectorizer_path: Path = BASE_DIR / "models" / "triage_vectorizer.joblib"

    tfidf_max_features: int = 20_000
    tfidf_ngram_max: int = 2

    reports_dir: Path = BASE_DIR / "reports"

    api_title: str = "Triagem de Laudos Médicos — API"
    use_onnx_runtime: bool = False
    """Se True, a API serve o classificador otimizado (ONNX Runtime)."""


def seed_everything(seed: int) -> None:
    """Fixa as seeds de todas as libs com aleatoriedade usadas no projeto."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


settings = Settings()
