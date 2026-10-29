"""Final evaluation: train on the whole train set and score once on djinni_test."""
import argparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from data.schema import weak_role
from models.constants import SPLITS, TFIDF_FIELDS, TRAIN, WEIGHT_PARAM
from models.metrics import evaluate
from models.tf_idf import build_model, inputs
from models.tracking import data_version, setup_mlflow


def main(field: str, djinni_weight: float) -> None:
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(SPLITS / "djinni_test.csv")

    weights = np.where(train["source"].eq("djinni"), djinni_weight, 1.0)
    model = build_model(field).fit(inputs(train, field), train["role"], **{WEIGHT_PARAM: weights})
    y_pred = model.predict(inputs(test, field))

    setup_mlflow()
    with mlflow.start_run(run_name=f"final-tfidf-{field}-w{djinni_weight:g}"):
        mlflow.log_params({
            "model": "tfidf + logistic regression",
            "field": field,
            "djinni_weight": djinni_weight,
            "train_rows": len(train),
            "test_rows": len(test),
            "data_splits_md5": data_version(),
        })
        metrics = evaluate(test["role"], y_pred, "test")
        metrics |= evaluate(test["role"], test["title"].map(weak_role), "rule_test")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")
        for prefix in ("test", "rule_test"):
            print(prefix, {k: round(v, 3) for k, v in metrics.items() if k.startswith(prefix) and ("macro" in k or "accuracy" in k)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("field", choices=TFIDF_FIELDS)
    parser.add_argument("--djinni-weight", type=float, default=1.0)
    args = parser.parse_args()
    main(args.field, args.djinni_weight)