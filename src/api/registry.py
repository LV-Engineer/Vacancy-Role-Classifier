"""Load the production models from the MLflow registry."""
from dataclasses import dataclass

import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from models.constants import MODEL_NAMES

@dataclass
class LoadedModel:
    pipeline: Pipeline
    field: str
    version: str

def load_production(target: str) -> LoadedModel:
    name = MODEL_NAMES[target]
    version = MlflowClient().get_model_version_by_alias(name, "production")
    pipeline = mlflow.sklearn.load_model(f"models:/{name}@production")
    return LoadedModel(pipeline=pipeline, field=version.tags["field"], version=version.version)