"""TF-IDF + Logistic Regression baseline: role from the title or from the description."""
import argparse

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer

from models.metrics import evaluate
from models.tracking import data_version, setup_mlflow
from models.constants import TFIDF_PARAMS, LOGREG_PARAMS, N_FOLDS, TRAIN, FIELDS, TFIDF_FIELDS
from models.cross_validation import cross_validate

def build_model(field: str) -> Pipeline:
    match field:
        case "both":
            features = make_column_transformer(
                (TfidfVectorizer(**TFIDF_PARAMS), "title"),
                (TfidfVectorizer(**TFIDF_PARAMS), "description"),
            )
        case "title" | "description":
            features = TfidfVectorizer(**TFIDF_PARAMS)
        case _:
            raise ValueError(f"unknown field: {field}")
    return make_pipeline(features, LogisticRegression(**LOGREG_PARAMS))

def inputs(train: pd.DataFrame, field: str) -> pd.Series | pd.DataFrame:
    columns = ["title", "description"] if field == "both" else field
    return train[columns].fillna("")

def main(field: str, djinni_weight: float) -> None:
    train = pd.read_csv(TRAIN)
    djinni, y_pred = cross_validate(train, inputs(train, field), lambda: build_model(field), djinni_weight)

    setup_mlflow()
    with mlflow.start_run(run_name=f"tfidf-{field}-w{djinni_weight:g}"):
        mlflow.log_params({
            "model": "tfidf + logistic regression",
            "field": field,
            "n_folds": N_FOLDS,
            "train_rows": len(train),
            "data_splits_md5": data_version(),
            "djinni_weight": djinni_weight,
            **{f"tfidf_{k}": v for k, v in TFIDF_PARAMS.items()},
            **{f"logreg_{k}": v for k, v in LOGREG_PARAMS.items()},
        })
        metrics = evaluate(djinni["role"], y_pred, "cv")
        mlflow.log_metrics(metrics)
        print(field, {k: round(v, 3) for k, v in metrics.items() if "macro" in k or "accuracy" in k})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("field", choices=TFIDF_FIELDS)
    parser.add_argument("--djinni-weight", type=float, default=1.0)
    args = parser.parse_args()
    main(args.field, args.djinni_weight)