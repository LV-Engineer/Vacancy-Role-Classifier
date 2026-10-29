"""TF-IDF + Logistic Regression baseline: role from the title or from the description."""
import sys

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline

from data.constants import SEED
from data.split import company_key
from models.metrics import evaluate
from models.tracking import data_version, setup_mlflow
from models.constants import TFIDF_PARAMS, LOGREG_PARAMS, N_FOLDS, TRAIN, FIELDS

def build_model() -> Pipeline:
    return make_pipeline(TfidfVectorizer(**TFIDF_PARAMS), LogisticRegression(**LOGREG_PARAMS))


def cross_validate(train: pd.DataFrame, field: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Predict the Djinni rows fold by fold: each fold is predicted by a model trained without it."""
    linkedin = train[train["source"] == "linkedin"]
    djinni = train[train["source"] == "djinni"].reset_index(drop=True)
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    groups = djinni["description"].map(company_key)

    y_pred = np.empty(len(djinni), dtype=object)
    for fit_idx, val_idx in folds.split(djinni, djinni["role"], groups):
        fit = pd.concat([linkedin, djinni.iloc[fit_idx]])
        model = build_model()
        model.fit(fit[field].fillna(""), fit["role"])
        y_pred[val_idx] = model.predict(djinni.iloc[val_idx][field].fillna(""))
    return djinni, y_pred


def main(field: str) -> None:
    train = pd.read_csv(TRAIN)
    djinni, y_pred = cross_validate(train, field)

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