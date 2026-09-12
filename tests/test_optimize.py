from triage.config import Settings
from triage.models.optimize import (
    benchmark_baseline,
    benchmark_onnx,
    export_classifier_to_onnx,
)


def test_export_and_onnx_benchmark_runs(tmp_path):
    settings = Settings(
        onnx_classifier_path=tmp_path / "clf.onnx",
        onnx_vectorizer_path=tmp_path / "vectorizer.joblib",
    )
    export_classifier_to_onnx(settings)
    assert settings.onnx_classifier_path.exists()
    assert settings.onnx_vectorizer_path.exists()

    sample_texts = [
        "Patient with chest pain and shortness of breath.",
        "Routine checkup, no complaints reported.",
    ]
    baseline_stats = benchmark_baseline(settings, sample_texts, n_requests=5)
    onnx_stats = benchmark_onnx(settings, sample_texts, n_requests=5)

    assert baseline_stats.n_requests == 5
    assert onnx_stats.n_requests == 5
    assert baseline_stats.mean_ms > 0
    assert onnx_stats.mean_ms > 0
