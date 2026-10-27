"""Rule baseline (no ML): role from title keywords. Logged to MLflow on Djinni dev and test."""

import re
from pathlib import Path

import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data.schema import ROLES, weak_role
from models.tracking import data_version, setup_mlflow

SPLITS = Path("data/splits")

def evaluate(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    y_true = df["role"]
    y_pred = df["title"].map(weak_role)

    metrics = {
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, labels=ROLES, average="macro", zero_division=0)),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
    }
    per_class = np.atleast_1d(f1_score(y_true, y_pred, labels=ROLES, average=None, zero_division=0))
    for role, score in zip(ROLES, per_class):
        metrics[f"{prefix}_f1_{re.sub(r'\W+', '_', role)}"] = float(score)

    confusion = pd.crosstab(y_true, y_pred).reindex(index=ROLES, columns=ROLES, fill_value=0)
    mlflow.log_text(confusion.to_csv(), f"{prefix}_confusion.csv")
    return metrics

def main() -> None:
    setup_mlflow()
    with mlflow.start_run(run_name="rule-baseline"):
        mlflow.log_params({"model": "title keyword rules", "data_splits_md5": data_version()})
        for name, prefix in (("djinni_dev", "dev"), ("djinni_test", "test")):
            df = pd.read_csv(SPLITS / f"{name}.csv")
            metrics = evaluate(df, prefix)
            mlflow.log_metrics(metrics)
            print(prefix, {k: round(v, 3) for k, v in metrics.items() if "macro" in k or "accuracy" in k})


if __name__ == "__main__":
    main()