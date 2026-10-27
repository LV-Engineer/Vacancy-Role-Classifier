"""MLflow run tracking shared by all training and evaluation scripts."""
import os
import re
from pathlib import Path

import mlflow

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "role-classifier"

def setup_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

def data_version(name: str = "splits") -> str:
    """Hash of a data folder from its DVC pointer, so a run is tied to the data it used."""
    pointer = Path("data") / f"{name}.dvc"
    found_match = re.search(r"md5: (\w+)", pointer.read_text(encoding="utf-8"))
    if found_match is None:
        raise ValueError(f"no md5 hash in {pointer}")
    return found_match.group(1)