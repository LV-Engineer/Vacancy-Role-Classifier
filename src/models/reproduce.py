"""Reproducibility check: re-train from a logged run and compare the test metrics."""
import argparse

import mlflow

from models.final_eval import train_and_score
from models.tracking import data_version, setup_mlflow

TOLERANCE = 1e-9
CHECKED = ("test_macro_f1", "test_accuracy", "rule_test_macro_f1", "rule_test_accuracy")

def main(run_id: str) -> None:
    setup_mlflow()
    original = mlflow.get_run(run_id)
    params = original.data.params
    if data_version() != params["data_splits_md5"]:
        raise SystemExit(
            f"data version differs ({data_version()} instead of {params['data_splits_md5']}): "
            "check out the run's commit and run `dvc checkout`"
        )

    with mlflow.start_run(run_name=f"reproduce-{run_id[:8]}"):
        _, metrics, _, _ = train_and_score(params.get("target", "role"), params["field"], float(params["djinni_weight"]))
        mlflow.log_params({"reproduced_run": run_id, "data_splits_md5": data_version()})
        mlflow.log_metrics(metrics)

    mismatches = {k: (original.data.metrics[k], metrics[k]) for k in CHECKED if abs(original.data.metrics[k] - metrics[k]) > TOLERANCE}
    if mismatches:
        raise SystemExit(f"NOT reproduced: {mismatches}")
    print(f"reproduced: {len(CHECKED)} metrics match run {run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    main(parser.parse_args().run_id)