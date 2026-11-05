import pytest
from fastapi.testclient import TestClient

from sentiment_mlops.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan so the model is loaded
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_positive_and_negative(client):
    pos = client.post("/v1/predict", json={"text": "Great phone, battery lasts all day. Love it!"}).json()
    neg = client.post("/v1/predict", json={"text": "Stopped working after a week. Waste of money."}).json()
    assert pos["label"] == "positive" and neg["label"] == "negative"
    assert 0 <= pos["positive_probability"] <= 1
    assert neg["top_terms"], "explanations should be returned by default"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_predict_rejects_empty(client, text):
    r = client.post("/v1/predict", json={"text": text})
    assert r.status_code == 422
    assert "empty" in r.json()["detail"].lower()


def test_batch_handles_empty_items_and_counts_concerns(client):
    texts = ["Love it", "", "Battery died and the seller never answered. Want a refund.", "Screen cracked, terrible"]
    body = client.post("/v1/predict/batch", json={"texts": texts}).json()
    assert body["n_total"] == 4 and body["n_scored"] == 3
    assert body["items"][1]["error"] == "empty text"
    counts = {c["key"]: c["negative_count"] for c in body["concerns"]}
    assert counts["battery"] >= 1 and counts["returns"] >= 1


def test_batch_limit(client):
    r = client.post("/v1/predict/batch", json={"texts": ["ok"] * 1001})
    assert r.status_code == 413


def test_model_info_and_metrics(client):
    info = client.get("/v1/model").json()
    assert info["metrics"]["macro_f1"] > 0.85
    assert client.get("/metrics").status_code == 200
    assert len(client.get("/v1/concerns").json()) == 10
