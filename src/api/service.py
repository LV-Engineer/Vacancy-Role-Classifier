"""Prediction logic of the service: no HTTP in here."""
import pandas as pd

from api.registry import LoadedModel
from api.schemas import Prediction, PredictResponse
from models.tf_idf import inputs

class PredictionService:
    def __init__(self, models: dict[str, LoadedModel]):
        self.models = models

    def predict(self, title: str, description: str) -> PredictResponse:
        frame = pd.DataFrame([{"title": title, "description": description}])
        return PredictResponse(role=self._predict_one("role", frame), level=self._predict_one("level", frame))

    def versions(self) -> dict[str, str]:
        return {target: model.version for target, model in self.models.items()}

    def _predict_one(self, target: str, frame: pd.DataFrame) -> Prediction:
        model = self.models[target]
        probabilities = model.pipeline.predict_proba(inputs(frame, model.field))[0]
        classes = [str(c) for c in model.pipeline.classes_]
        best = int(probabilities.argmax())
        return Prediction(
            label=classes[best],
            probability=float(probabilities[best]),
            probabilities={c: float(p) for c, p in zip(classes, probabilities)}
        )