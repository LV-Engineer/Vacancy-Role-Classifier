import numpy as np
import pytest

from api.registry import LoadedModel
from api.service import PredictionService

class FakePipeline:
    def __init__(self, classes: list[str], probabilities: list[float]):
        self.classes_ = np.array(classes)
        self._probabilities = np.array([probabilities])

    def predict_proba(self, X):
        return self._probabilities

@pytest.fixture
def service() -> PredictionService:
    return PredictionService({
        "role": LoadedModel(FakePipeline(["Data Analyst", "Other"], [0.8, 0.2]), "title", "7"),
        "level": LoadedModel(FakePipeline(["junior", "not_junior"], [0.3, 0.7]), "both", "3"),
    })