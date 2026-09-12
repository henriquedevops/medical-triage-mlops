"""Baselines simples para contextualizar o ganho do modelo principal.

Mesma lógica da Fase 2 (comparar o modelo "de verdade" com baselines
ingênuos): aqui usamos (1) classe majoritária e (2) Naive Bayes multinomial
sobre a mesma matriz TF-IDF, para deixar claro o quanto de sinal a
Regressão Logística está de fato aprendendo.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from triage.config import Settings


def _majority_baseline(settings: Settings) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=settings.tfidf_max_features)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )


def _naive_bayes_baseline(settings: Settings) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=settings.tfidf_max_features,
                    ngram_range=(1, settings.tfidf_ngram_max),
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )


def compare_baselines(
    settings: Settings,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    main_model_macro_f1: float,
) -> dict[str, float]:
    """Treina os baselines e retorna o macro-F1 de cada um, com o modelo principal."""
    results = {"logistic_regression_tfidf": main_model_macro_f1}

    for name, factory in (
        ("majority_class", _majority_baseline),
        ("naive_bayes_tfidf", _naive_bayes_baseline),
    ):
        pipeline = factory(settings)
        pipeline.fit(train_df["medical_abstract"], train_df["urgency"])
        preds = pipeline.predict(test_df["medical_abstract"])
        results[name] = f1_score(test_df["urgency"], preds, average="macro")

    return results
