"""Classification metrics logged to MLflow, shared by all model scripts."""
import re

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate(y_true, y_pred, prefix: str, labels: list[str]) -> dict[str, float]:
    metrics = {
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
    }
    per_class = np.atleast_1d(f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0))
    for role, score in zip(labels, per_class):
        metrics[f"{prefix}_f1_{re.sub(r'\W+', '_', role)}"] = float(score)

    confusion = pd.crosstab(y_true, y_pred).reindex(index=labels, columns=labels, fill_value=0)
    mlflow.log_text(confusion.to_csv(), f"{prefix}_confusion.csv")
    return metrics