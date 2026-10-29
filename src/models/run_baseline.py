"""Rule baseline (no ML): role from title keywords. Logged to MLflow on Djinni dev and test."""
import mlflow
import pandas as pd
import numpy as np

from data.schema import weak_role
from models.tracking import data_version, setup_mlflow
from models.metrics import evaluate
from models.constants import SPLITS

def main() -> None:
    setup_mlflow()
    with mlflow.start_run(run_name="rule-baseline"):
        mlflow.log_params({"model": "title keyword rules", "data_splits_md5": data_version()})
        for name, prefix in (("djinni_dev", "dev"), ("djinni_test", "test")):
            df = pd.read_csv(SPLITS / f"{name}.csv")
            metrics = evaluate(df["role"], df["title"].map(weak_role), prefix)
            mlflow.log_metrics(metrics)
            print(prefix, {k: round(v, 3) for k, v in metrics.items() if "macro" in k or "accuracy" in k})


if __name__ == "__main__":
    main()