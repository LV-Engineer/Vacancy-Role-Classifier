"""Register a trained model and promote it to production if it passes the quality gate."""
import argparse

import mlflow
from mlflow import MlflowClient

from models.constants import MODEL_NAME, PROMOTION_METRIC, PROMOTION_THRESHOLD
from models.tracking import setup_mlflow


def main(run_id: str) -> None:
    setup_mlflow()
    client = MlflowClient()
    score = client.get_run(run_id).data.metrics[PROMOTION_METRIC]

    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "candidate", version.version)
    client.set_model_version_tag(MODEL_NAME, version.version, PROMOTION_METRIC, f"{score:.3f}")

    if score >= PROMOTION_THRESHOLD:
        client.set_registered_model_alias(MODEL_NAME, "production", version.version)
        print(f"v{version.version}: {PROMOTION_METRIC} {score:.3f} >= {PROMOTION_THRESHOLD}, promoted to production")
    else:
        print(f"v{version.version}: {PROMOTION_METRIC} {score:.3f} < {PROMOTION_THRESHOLD}, stays candidate")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    main(parser.parse_args().run_id)