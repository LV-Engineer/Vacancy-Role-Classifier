"""Error analysis: where a model is wrong on Djinni, compared with the title rule."""
import argparse

import mlflow
import pandas as pd
from sklearn.metrics import classification_report

from data.constants import ROLES, SEED
from rules.roles import weak_role
from models import embeddings, tf_idf
from models.constants import TFIDF_FIELDS, N_SHOWN, TRAIN
from models.cross_validation import cross_validate
from models.tracking import setup_mlflow

MODELS = {
    "tfidf": lambda train, field: (tf_idf.inputs(train, field), lambda: tf_idf.build_model(field)),
    "embeddings": lambda train, field: (embeddings.embed(train[field]), embeddings.build_model),
}

def main(model: str, field: str, djinni_weight: float) -> None:
    train = pd.read_csv(TRAIN)
    X, build_model = MODELS[model](train, field)
    djinni, y_pred = cross_validate(train, X, build_model, djinni_weight)
    djinni = djinni.assign(predicted=y_pred, rule=djinni["title"].map(weak_role))
    djinni["wrong"] = djinni["role"] != djinni["predicted"]
    wrong = djinni[djinni["wrong"]]

    confusion = pd.crosstab(djinni["role"], djinni["predicted"]).reindex(index=ROLES, columns=ROLES, fill_value=0)
    print(classification_report(djinni["role"], y_pred, labels=ROLES, zero_division=0))
    print(confusion, end="\n\n")
    print("error rate by language:", djinni.groupby("language")["wrong"].mean().round(3).to_dict())
    print("model wrong, rule right:", int((wrong["rule"] == wrong["role"]).sum()), "of", len(wrong))
    print("model right, rule wrong:", int(((djinni["role"] == djinni["predicted"]) & (djinni["rule"] != djinni["role"])).sum()), end="\n\n")

    columns = ["id", "title", "role", "predicted", "rule"]
    sample = wrong.sample(min(N_SHOWN, len(wrong)), random_state=SEED)[columns]
    print(sample.to_string(index=False, max_colwidth=60))

    setup_mlflow()
    with mlflow.start_run(run_name=f"error-analysis-{model}-{field}-w{djinni_weight:g}"):
        mlflow.log_params({"model": model, "field": field, "djinni_weight": djinni_weight})
        mlflow.log_text(confusion.to_csv(), "confusion.csv")
        mlflow.log_text(wrong[columns + ["description"]].to_csv(index=False), "errors.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("field", choices=TFIDF_FIELDS)
    parser.add_argument("--djinni-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.model == "embeddings" and args.field == "both":
        parser.error("embeddings support only title or description")
    main(args.model, args.field, args.djinni_weight)