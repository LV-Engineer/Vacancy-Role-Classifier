"""Load the production models from the MLflow registry."""
from dataclasses import dataclass
from typing import Any, Protocol

import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

from models.constants import MODEL_NAMES

class Classifier(Protocol):
    """What the service needs from a model: a fitted sklearn pipeline satisfies it."""
    classes_: np.ndarray

    def predict_proba(self, X: Any) -> np.ndarray: ...

@dataclass
class LoadedModel:
    pipeline: Classifier
    field: str
    version: str

def load_production(target: str) -> LoadedModel:
    name = MODEL_NAMES[target]
    version = MlflowClient().get_model_version_by_alias(name, "production")
    pipeline = mlflow.sklearn.load_model(f"models:/{name}@production")
    return LoadedModel(pipeline=pipeline, field=version.tags["field"], version=version.version)