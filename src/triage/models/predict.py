"""Wrapper de inferência único, usado pela API.

Abstrai qual backend está servindo o classificador (pipeline scikit-learn
completo, ou vetorizador scikit-learn + classificador ONNX Runtime) atrás de
uma única interface `predict(text) -> (urgency, scores)`, controlada por
`settings.use_onnx_runtime`.
"""

from __future__ import annotations

import numpy as np

from triage.config import Settings
from triage.data.labels import URGENCY_LEVELS


class TriagePredictor:
    """Carrega o(s) artefato(s) de modelo uma única vez e serve predições."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._backend = "onnx" if self.settings.use_onnx_runtime else "sklearn"

        if self._backend == "onnx":
            self._load_onnx_backend()
        else:
            self._load_sklearn_backend()

    def _load_sklearn_backend(self) -> None:
        import joblib

        self._pipeline = joblib.load(self.settings.baseline_model_path)

    def _load_onnx_backend(self) -> None:
        import joblib
        import onnxruntime as rt

        self._vectorizer = joblib.load(self.settings.onnx_vectorizer_path)
        self._session = rt.InferenceSession(
            str(self.settings.onnx_classifier_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        # Ordem das classes tal como vista pelo classificador original.
        self._classes = list(self._pipeline_classes())

    def _pipeline_classes(self) -> list[str]:
        import joblib

        pipeline = joblib.load(self.settings.baseline_model_path)
        return list(pipeline.named_steps["clf"].classes_)

    @property
    def backend(self) -> str:
        return self._backend

    def predict(self, text: str) -> dict:
        """Retorna o nível de urgência previsto e as probabilidades por classe."""
        if self._backend == "sklearn":
            return self._predict_sklearn(text)
        return self._predict_onnx(text)

    def _predict_sklearn(self, text: str) -> dict:
        proba = self._pipeline.predict_proba([text])[0]
        classes = list(self._pipeline.named_steps["clf"].classes_)
        return self._format_result(classes, proba)

    def _predict_onnx(self, text: str) -> dict:
        vec = self._vectorizer.transform([text]).toarray().astype(np.float32)
        _, proba_list = self._session.run(None, {self._input_name: vec})
        proba_map = proba_list[0]
        classes = list(proba_map.keys())
        proba = [proba_map[c] for c in classes]
        return self._format_result(classes, proba)

    @staticmethod
    def _format_result(classes: list[str], proba) -> dict:
        scores = {cls: float(p) for cls, p in zip(classes, proba, strict=True)}
        predicted = max(scores, key=scores.get)
        # Garante que toda classe conhecida apareça na resposta, mesmo com prob. 0.
        for level in URGENCY_LEVELS:
            scores.setdefault(level, 0.0)
        return {"urgency": predicted, "scores": scores}
