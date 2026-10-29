"""Final evaluation: train on the whole train set and score once on djinni_test."""
import argparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from models.constants import SPLITS, TFIDF_FIELDS, TRAIN, WEIGHT_PARAM, LABELS, TARGETS
from rules.by_target import RULES
from models.metrics import evaluate
from models.tf_idf import build_model, inputs
from models.tracking import data_version, setup_mlflow

def train_and_score(target: str, field: str, djinni_weight: float):
    """Fit on the whole train set and score on djinni_test; logs the confusion matrices to the active run."""
    train = pd.read_csv(TRAIN).dropna(subset=[target])
    test = pd.read_csv(SPLITS / "djinni_test.csv")

    weights = np.where(train["source"].eq("djinni"), djinni_weight, 1.0)
    model = build_model(field).fit(inputs(train, field), train[target], **{WEIGHT_PARAM: weights})
    y_pred = model.predict(inputs(test, field))

    metrics = evaluate(test[target], y_pred, "test", LABELS[target])  
    metrics |= evaluate(test[target], test["title"].map(RULES[target]), "rule_test", LABELS[target])
    return model, metrics, len(train), len(test)

def main(target: str, field: str, djinni_weight: float) -> None:
    setup_mlflow()
    with mlflow.start_run(run_name=f"final-{target}-tfidf-{field}-w{djinni_weight:g}"):
        model, metrics, train_rows, test_rows = train_and_score(target, field, djinni_weight)
        mlflow.log_params({
            "model": "tfidf + logistic regression",
            "target": target,
            "field": field,
            "djinni_weight": djinni_weight,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "data_splits_md5": data_version(),
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")
        for prefix in ("test", "rule_test"):
            print(prefix, {k: round(v, 3) for k, v in metrics.items() if k.startswith(prefix) and ("macro" in k or "accuracy" in k)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("field", choices=TFIDF_FIELDS)
    parser.add_argument("--target", choices=TARGETS, default="role")
    parser.add_argument("--djinni-weight", type=float, default=1.0)
    args = parser.parse_args()
    main(args.target, args.field, args.djinni_weight)