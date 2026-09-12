"""Carregamento do Medical Abstracts TC Corpus e derivação do rótulo de urgência."""

from pathlib import Path

import pandas as pd

from triage.data.labels import condition_to_urgency


def load_raw_split(csv_path: Path) -> pd.DataFrame:
    """Lê um split bruto (train/test) do corpus.

    Espera colunas `condition_label` (int, 1-5) e `medical_abstract` (str).
    """
    df = pd.read_csv(csv_path)
    expected_cols = {"condition_label", "medical_abstract"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {csv_path}: {missing}")
    return df


def add_urgency_label(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a coluna `urgency` derivada de `condition_label`."""
    out = df.copy()
    out["urgency"] = out["condition_label"].map(condition_to_urgency)
    out["medical_abstract"] = out["medical_abstract"].str.strip()
    return out


def load_dataset(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega e prepara os splits de treino e teste com a coluna `urgency`."""
    train_df = add_urgency_label(load_raw_split(train_path))
    test_df = add_urgency_label(load_raw_split(test_path))
    return train_df, test_df
