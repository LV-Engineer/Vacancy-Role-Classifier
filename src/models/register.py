"""Register a trained model and promote it to production if it passes the quality gate."""
import argparse

import mlflow
from mlflow import MlflowClient

from models.constants import MODEL_NAMES, PROMOTION_METRIC, PROMOTION_TOLERANCE
from models.tracking import setup_mlflow


def main(run_id: str) -> None:
    setup_mlflow()
    client = MlflowClient()
    run = client.get_run(run_id)
    name = MODEL_NAMES[run.data.params.get("target", "role")]
    score = run.data.metrics[PROMOTION_METRIC]
    threshold = run.data.metrics["rule_test_macro_f1"] - PROMOTION_TOLERANCE

    version = mlflow.register_model(f"runs:/{run_id}/model", name)
    client.set_registered_model_alias(name, "candidate", version.version)
    client.set_model_version_tag(name, version.version, PROMOTION_METRIC, f"{score:.3f}")
    client.set_model_version_tag(name, version.version, "field", run.data.params["field"])

    if score >= threshold:
        client.set_registered_model_alias(name, "production", version.version)
        print(f"v{version.version}: {PROMOTION_METRIC} {score:.3f} >= {threshold:.3f}, promoted to production")
    else:
        print(f"v{version.version}: {PROMOTION_METRIC} {score:.3f} < {threshold:.3f}, stays candidate")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    main(parser.parse_args().run_id)