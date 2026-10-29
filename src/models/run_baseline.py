"""Rule baseline (no ML): role from title keywords. Logged to MLflow on Djinni dev and test."""
import argparse

import mlflow
import pandas as pd

from models.constants import LABELS, SPLITS, TARGETS
from models.metrics import evaluate
from models.tracking import data_version, setup_mlflow
from rules.by_target import RULES

def main(target: str) -> None:
    setup_mlflow()
    with mlflow.start_run(run_name=f"rule-baseline-{target}"):
        mlflow.log_params({"model": "title keyword rules", "target": target, "data_splits_md5": data_version()})
        for name, prefix in (("djinni_dev", "dev"), ("djinni_test", "test")):
            df = pd.read_csv(SPLITS / f"{name}.csv")
            metrics = evaluate(df[target], df["title"].map(RULES[target]), prefix, LABELS[target])
            mlflow.log_metrics(metrics)
            print(prefix, {k: round(v, 3) for k, v in metrics.items() if "macro" in k or "accuracy" in k})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=TARGETS, default="role")
    main(parser.parse_args().target)