import json
import os
from pathlib import Path

import numpy as np
import pytest

SERVING = Path(os.environ.get("SERVING_DIR", "data/serving"))
pytestmark = pytest.mark.skipif(not (SERVING / "model.txt").exists(),
                                reason="fichiers de l'API absents : lancer src/export_serving.py")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_zones(client):
    expected = len(json.loads((SERVING / "zones.json").read_text()))
    assert len(client.get("/zones").json()) == expected

def test_forecast_24_heures(client):
    r = client.get("/forecast", params={"zone_id": 132, "date": "2025-07-15"})
    assert r.status_code == 200
    hours = r.json()["hours"]
    assert len(hours) == 24 and all(h["forecast"] >= 0 for h in hours)


def test_forecast_identique_au_modele_final(client):
    """L'API doit donner exactement les prévisions du modèle final, calculées hors de l'API."""
    model_path = Path("models/lgbm_final.txt")
    if SERVING != Path("data/serving") or not model_path.exists():
        pytest.skip("comparaison possible seulement avec les vraies données servies")
    import lightgbm as lgb

    from eval_lib import load_dataset

    df = load_dataset()
    df["pu_zone"] = df["pu_zone"].astype(int).astype("category")
    day = df[(df["pu_zone"] == 132) & (df["date"] == "2025-11-15")].sort_values("hour")
    booster = lgb.Booster(model_file=str(model_path))
    expected = booster.predict(day[booster.feature_name()]).clip(min=0)

    api = client.get("/forecast", params={"zone_id": 132, "date": "2025-11-15"}).json()["hours"]
    assert np.allclose([h["forecast"] for h in api], expected, atol=0.01)

def test_zone_inconnue(client):
    assert client.get("/forecast", params={"zone_id": 999, "date": "2025-07-15"}).status_code == 404


def test_date_hors_periode(client):
    assert client.get("/forecast", params={"zone_id": 132, "date": "2030-01-01"}).status_code == 404


def test_date_mal_formee(client):
    assert client.get("/forecast", params={"zone_id": 132, "date": "hier"}).status_code == 422