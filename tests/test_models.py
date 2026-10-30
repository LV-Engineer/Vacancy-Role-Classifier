import mlflow
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data.constants import LEVELS, ROLES
from models.cross_validation import cross_validate
from models.metrics import evaluate
from models.tf_idf import build_model, inputs
from models import tf_idf

def make_train() -> pd.DataFrame:
    linkedin = pd.DataFrame({
        "source": "linkedin",
        "title": ["alpha job"] * 20 + ["beta job"] * 20,
        "description": [f"linkedin text {i}" for i in range(40)],
        "role": ["Data Analyst"] * 20 + ["Other"] * 20,
        "level": None,
    })
    djinni = pd.DataFrame({
        "source": "djinni",
        "title": ["alpha engineer"] * 15 + ["beta engineer"] * 15,
        "description": [f"djinni company number {i} writes unique text" for i in range(30)],
        "role": ["Data Analyst"] * 15 + ["Other"] * 15,
        "level": ["junior"] * 15 + ["not_junior"] * 15,
    })
    return pd.concat([linkedin, djinni], ignore_index=True)

def simple_model():
    return make_pipeline(CountVectorizer(), LogisticRegression())

def test_cross_validate_predicts_every_djinni_row_once():
    train = make_train()
    djinni, y_pred = cross_validate(train, train["title"], simple_model)
    assert len(djinni) == len(y_pred) == 30
    assert (y_pred == djinni["role"].to_numpy()).mean() > 0.9

def test_cross_validate_skips_rows_without_a_label():
    train = make_train()
    djinni, y_pred = cross_validate(train, train["title"], simple_model, target="level")
    assert set(y_pred) <= set(LEVELS)

def test_evaluate_perfect_predictions(monkeypatch):
    logged = []
    monkeypatch.setattr(mlflow, "log_text", lambda text, artifact_file: logged.append(artifact_file))
    metrics = evaluate(ROLES, ROLES, "cv", ROLES)
    assert metrics["cv_macro_f1"] == 1.0
    assert metrics["cv_accuracy"] == 1.0
    assert metrics["cv_f1_ML_AI_Engineer"] == 1.0
    assert logged == ["cv_confusion.csv"]

def test_evaluate_all_wrong(monkeypatch):
    monkeypatch.setattr(mlflow, "log_text", lambda text, artifact_file: None)
    metrics = evaluate(["junior", "not_junior"], ["not_junior", "junior"], "cv", LEVELS)
    assert metrics["cv_macro_f1"] == 0.0
    assert metrics["cv_accuracy"] == 0.0

def test_inputs_for_one_field_and_for_both():
    frame = pd.DataFrame({"title": ["a", None], "description": ["b", "c"]})
    assert inputs(frame, "title").tolist() == ["a", ""]
    assert list(inputs(frame, "both").columns) == ["title", "description"]

def test_build_model_rejects_an_unknown_field():
    with pytest.raises(ValueError):
        build_model("salary")

@pytest.mark.parametrize("field", ["title", "description", "both"])
def test_build_model_learns_a_simple_rule(field):
    frame = pd.DataFrame({
        "title": ["data analyst", "python backend"] * 3,
        "description": ["sql bi", "django api"] * 3,
    })
    labels = ["Data Analyst", "Python Backend"] * 3
    model = build_model(field).fit(inputs(frame, field), labels)
    assert list(model.predict(inputs(frame, field)))[:2] == ["Data Analyst", "Python Backend"]

@pytest.fixture
def mlflow_store(tmp_path):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield
    mlflow.set_tracking_uri(previous)

def test_tfidf_main_logs_a_run(tmp_path, monkeypatch, mlflow_store):
    train_csv = tmp_path / "train.csv"
    make_train().to_csv(train_csv, index=False)
    monkeypatch.setattr(tf_idf, "TRAIN", train_csv)
    monkeypatch.setattr(tf_idf, "setup_mlflow", lambda: None)
    monkeypatch.setattr(tf_idf, "data_version", lambda: "test-md5")

    tf_idf.main("role", "title", 1.0)

    runs = mlflow.search_runs(search_all_experiments=True)
    assert runs["params.field"].tolist() == ["title"]
    assert runs["params.data_splits_md5"].tolist() == ["test-md5"]
    assert runs["metrics.cv_accuracy"].iloc[0] > 0.9