.PHONY: install lint test train optimize serve compose-up compose-down dag-test

install:
	uv sync

lint:
	uv run ruff check src/ tests/

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

train:
	uv run python -m triage.models.train

optimize:
	uv run python -m triage.models.optimize

serve:
	uv run uvicorn triage.api.main:app --reload

compose-up:
	docker compose up --build

compose-down:
	docker compose down

dag-test:
	AIRFLOW_HOME=/tmp/airflow_home PYTHONPATH=src \
		airflow dags test medical_triage_retrain 2026-01-01
