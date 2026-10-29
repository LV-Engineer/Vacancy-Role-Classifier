"""TF-IDF + Logistic Regression baseline: role from the title or from the description."""
import sys

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from models.metrics import evaluate
from models.tracking import data_version, setup_mlflow
from models.constants import TFIDF_PARAMS, LOGREG_PARAMS, N_FOLDS, TRAIN, FIELDS
from models.cross_validation import cross_validate

def build_model() -> Pipeline:
    return make_pipeline(TfidfVectorizer(**TFIDF_PARAMS), LogisticRegression(**LOGREG_PARAMS))

def main(field: str) -> None:
    train = pd.read_csv(TRAIN)
    djinni, y_pred = cross_validate(train, train[field].fillna(""), build_model)

    setup_mlflow()
    with mlflow.start_run(run_name=f'tfidf-{field}'):
        mlflow.log_params({
            "model": "tfidf + logistic regression",
            "field": field,
            "n_folds": N_FOLDS,
            "train_rows": len(train),
            "data_splits_md5": data_version(),
            **{f"tfidf_{k}": v for k, v in TFIDF_PARAMS.items()},
            **{f"logreg_{k}": v for k, v in LOGREG_PARAMS.items()},
        })
        metrics = evaluate(djinni["role"], y_pred, "cv")
        mlflow.log_metrics(metrics)
        print(field, {k: round(v, 3) for k, v in metrics.items() if "macro" in k or "accuracy" in k})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in FIELDS:
        raise SystemExit(f"usage: tfidf.py {' | '.join(FIELDS)}")
    main(sys.argv[1])