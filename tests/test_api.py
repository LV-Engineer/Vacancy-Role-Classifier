import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import get_service
from api import registry

@pytest.fixture
def client(service):
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()

def test_service_picks_the_best_class(service):
    response = service.predict("Data Analyst", "SQL")
    assert response.role.label == "Data Analyst"
    assert response.role.probability == pytest.approx(0.8)
    assert response.level.label == "not_junior"
    assert sum(response.level.probabilities.values()) == pytest.approx(1.0)

def test_health_shows_model_versions(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": {"role": "7", "level": "3"}}

def test_predict(client):
    response = client.post("/predict", json={"title": "Data Analyst", "description": "SQL"})
    assert response.status_code == 200
    assert response.json()["role"]["label"] == "Data Analyst"

def test_predict_without_description_is_allowed(client):
    assert client.post("/predict", json={"title": "Data Analyst"}).status_code == 200

def test_predict_rejects_an_empty_title(client):
    assert client.post("/predict", json={"title": ""}).status_code == 422

def test_app_loads_models_on_startup(monkeypatch, service):
    monkeypatch.setattr("api.main.setup_mlflow", lambda: None)
    monkeypatch.setattr("api.main.load_production", lambda target: service.models[target])
    with TestClient(app) as client:
        assert client.get("/health").json()["models"] == {"role": "7", "level": "3"}

def test_load_production_reads_the_field_tag(monkeypatch):
    class Version:
        tags = {"field": "title"}
        version = "5"

    class Client:
        def get_model_version_by_alias(self, name, alias):
            assert alias == "production"
            return Version()

    monkeypatch.setattr(registry, "MlflowClient", Client)
    monkeypatch.setattr(registry.mlflow.sklearn, "load_model", lambda uri: "pipeline")
    loaded = registry.load_production("role")
    assert (loaded.pipeline, loaded.field, loaded.version) == ("pipeline", "title", "5")