"""Sentence embeddings + Logistic Regression: role from the title or from the description."""
import sys
import argparse

import mlflow
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import make_pipeline

from models.constants import EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL, FIELDS, LOGREG_PARAMS, N_FOLDS, TRAIN
from models.cross_validation import cross_validate
from sklearn.linear_model import LogisticRegression
from models.tracking import data_version, setup_mlflow
from models.metrics import evaluate

def build_model():
    return make_pipeline(LogisticRegression(**LOGREG_PARAMS))

def embed(texts: pd.Series) -> np.ndarray:
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    encoder.max_seq_length = EMBEDDING_MAX_TOKENS
    return encoder.encode(
        ["query: " + text for text in texts.fillna("")],
        batch_size=32, normalize_embeddings=True, show_progress_bar=True,
    )

def main(field: str, djinni_weight: float) -> None:
    train = pd.read_csv(TRAIN)
    djinni, y_pred = cross_validate(train, embed(train[field]), build_model, djinni_weight)

    setup_mlflow()
    with mlflow.start_run(run_name=f"embeddings-{field}-w{djinni_weight:g}"):
        mlflow.log_params({
            "model": "embeddings + logistic regression",
            "field": field,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_max_tokens": EMBEDDING_MAX_TOKENS,
            "n_folds": N_FOLDS,
            "train_rows": len(train),
            "data_splits_md5": data_version(),
            "djinni_weight": djinni_weight,
            **{f"logreg_{k}": v for k, v in LOGREG_PARAMS.items()},
        })
        metrics = evaluate(djinni["role"], y_pred, "cv")
        mlflow.log_metrics(metrics)
        print(field, {k: round(v, 3) for k, v in metrics.items() if "macro" in k or "accuracy" in k})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("field", choices=FIELDS)
    parser.add_argument("--djinni-weight", type=float, default=1.0)
    args = parser.parse_args()
    main(args.field, args.djinni_weight)