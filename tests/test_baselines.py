from triage.config import Settings
from triage.data.loader import load_dataset
from triage.models.baselines import compare_baselines
from triage.models.train import train_and_evaluate


def test_compare_baselines_ranks_main_model_above_majority_class():
    settings = Settings()
    train_df, test_df = load_dataset(settings.raw_train_path, settings.raw_test_path)
    # Usa uma amostra para manter o teste rápido (o corpus completo é validado
    # manualmente via `python -m triage.models.train`).
    train_sample = train_df.sample(n=800, random_state=settings.seed)
    test_sample = test_df.sample(n=200, random_state=settings.seed)

    result = train_and_evaluate(settings, train_sample, test_sample)
    comparison = compare_baselines(
        settings, train_sample, test_sample, result.metrics["macro_f1"]
    )

    assert comparison["logistic_regression_tfidf"] > comparison["majority_class"]
    assert set(comparison.keys()) == {
        "logistic_regression_tfidf",
        "majority_class",
        "naive_bayes_tfidf",
    }
