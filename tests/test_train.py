import pandas as pd

from triage.config import Settings
from triage.data.loader import add_urgency_label
from triage.models.train import build_pipeline, train_and_evaluate


def _toy_dataframe(n_per_class: int = 15) -> pd.DataFrame:
    """Dataset sintético minúsculo só para exercitar o pipeline de treino nos testes.

    Usa vocabulário claramente separável por classe para garantir uma acurácia alta
    e testes rápidos (o dataset real, de ~14k amostras, é validado manualmente via
    `python -m triage.models.train`, não a cada `pytest`).
    """
    texts, labels = [], []
    vocab = {
        1: "tumor neoplasia cancer malignant biopsy",
        2: "digestive stomach bowel intestine ulcer",
        3: "nervous brain neuron seizure epilepsy",
        4: "cardiovascular heart artery infarction cardiac",
        5: "pathological general inflammation lesion syndrome",
    }
    for label, words in vocab.items():
        for i in range(n_per_class):
            texts.append(f"{words} case number {i} report")
            labels.append(label)
    df = pd.DataFrame({"condition_label": labels, "medical_abstract": texts})
    return add_urgency_label(df)


def test_build_pipeline_has_tfidf_and_classifier_steps():
    pipeline = build_pipeline(Settings())
    assert list(pipeline.named_steps.keys()) == ["tfidf", "clf"]


def test_train_and_evaluate_learns_something_on_toy_data():
    settings = Settings(tfidf_max_features=500)
    df = _toy_dataframe()
    train_df = df.sample(frac=0.8, random_state=settings.seed)
    test_df = df.drop(train_df.index)

    result = train_and_evaluate(settings, train_df, test_df)

    assert 0.0 <= result.metrics["macro_f1"] <= 1.0
    # Com vocabulário bem separável por classe, o modelo deve aprender razoavelmente.
    assert result.metrics["macro_f1"] > 0.5
