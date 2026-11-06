import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
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


def test_forecast_identique_a_la_validation(client):
    saved_path = Path("data/processed/val_predictions_lgbm.parquet")
    if SERVING != Path("data/serving") or not saved_path.exists():
        pytest.skip("comparaison possible seulement avec les vraies données servies")
    saved = pd.read_parquet(saved_path)
    saved["pu_zone"] = saved["pu_zone"].astype(int)
    saved = saved[(saved["pu_zone"] == 132)
                  & (pd.to_datetime(saved["date"]) == "2025-07-15")].sort_values("hour")
    api = client.get("/forecast", params={"zone_id": 132, "date": "2025-07-15"}).json()["hours"]
    assert np.allclose([h["forecast"] for h in api], saved["pred"].to_numpy(), atol=0.01)


def test_zone_inconnue(client):
    assert client.get("/forecast", params={"zone_id": 999, "date": "2025-07-15"}).status_code == 404


def test_date_hors_periode(client):
    assert client.get("/forecast", params={"zone_id": 132, "date": "2030-01-01"}).status_code == 404


def test_date_mal_formee(client):
    assert client.get("/forecast", params={"zone_id": 132, "date": "hier"}).status_code == 422