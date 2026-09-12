"""Fixtures compartilhadas.

Garante que exista um modelo treinado em disco antes da suíte rodar — assim
`pytest` funciona sozinho (sem exigir um passo manual de treino antes),
tanto localmente quanto no workflow de CI/CD.
"""

import pytest

from triage.config import Settings
from triage.models.train import run_training_job


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model() -> None:
    settings = Settings()
    if not settings.baseline_model_path.exists():
        run_training_job(settings)
