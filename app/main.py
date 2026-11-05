"""API de prévision de la demande de taxis à New York (étape 8)."""
import json
import os
from datetime import date, datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DATA = Path(os.environ.get("SERVING_DIR", "data/serving"))
MODEL_NAME = "LightGBM, perte de Poisson, sans météo"

# ---------- Chargement, une seule fois au démarrage ----------
booster = lgb.Booster(model_file=str(DATA / "model.txt"))
FEATURE_NAMES = booster.feature_name()          # même ordre qu'à l'entraînement
features = pd.read_parquet(DATA / "features.parquet")
features["pu_zone"] = features["pu_zone"].astype(int)
features["date"] = pd.to_datetime(features["date"]).dt.date
ZONES = json.loads((DATA / "zones.json").read_text())
ZONE_BY_ID = {z["id"]: z for z in ZONES}
MIN_DATE, MAX_DATE = features["date"].min(), features["date"].max()
EPISODES = pd.read_csv(DATA / "anomaly_episodes.csv", parse_dates=["debut", "fin"])


# ---------- Forme des réponses ----------
class Zone(BaseModel):
    id: int
    name: str
    borough: str


class HourForecast(BaseModel):
    hour: datetime
    forecast: float
    baseline_4w: float
    actual: int | None


class Forecast(BaseModel):
    zone_id: int
    zone_name: str
    date: date
    model: str
    total_forecast: float
    hours: list[HourForecast]


class Episode(BaseModel):
    start: datetime
    end: datetime
    hours: int
    score: float
    excess_trips: float
    zones_flagged: int


app = FastAPI(title="Prévision de la demande de taxis à New York", version="1.0.0",
              description="Prévisions horaires des prises en charge pour 56 zones, et épisodes anormaux.")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "zones": len(ZONES),
            "dates": [str(MIN_DATE), str(MAX_DATE)]}


@app.get("/zones", response_model=list[Zone])
def list_zones():
    return ZONES


@app.get("/forecast", response_model=Forecast)
def forecast(zone_id: int = Query(..., description="Numéro de zone TLC (voir /zones)"),
             day: date = Query(..., alias="date", description="Jour à prévoir, AAAA-MM-JJ")):
    if zone_id not in ZONE_BY_ID:
        raise HTTPException(status_code=404, detail=f"Zone {zone_id} non servie : voir /zones")
    if not MIN_DATE <= day <= MAX_DATE:
        raise HTTPException(status_code=404, detail=f"Date hors période servie ({MIN_DATE} au {MAX_DATE})")

    rows = features[(features["pu_zone"] == zone_id) & (features["date"] == day)].sort_values("hour")
    X = rows[FEATURE_NAMES].copy()
    X["pu_zone"] = X["pu_zone"].astype("category")   # comme à l'entraînement
    preds = booster.predict(X).clip(min=0)

    hours = [HourForecast(hour=h, forecast=round(float(p), 2), baseline_4w=round(float(b), 2),
                          actual=None if pd.isna(a) else int(a))
             for h, p, b, a in zip(rows["hour"], preds, rows["mean_same_slot_4w"], rows["n_pickups"])]
    return Forecast(zone_id=zone_id, zone_name=ZONE_BY_ID[zone_id]["name"], date=day,
                    model=MODEL_NAME, total_forecast=round(float(preds.sum()), 1), hours=hours)


@app.get("/anomalies/episodes", response_model=list[Episode])
def anomaly_episodes(limit: int = Query(10, ge=1, le=50, description="Nombre d'épisodes, les plus forts d'abord")):
    return [Episode(start=r.debut, end=r.fin, hours=int(r.heures), score=round(float(r.z_max), 1),
                    excess_trips=round(float(r.ecart_trajets), 1), zones_flagged=int(r.zones_signalees))
            for r in EPISODES.head(limit).itertuples()]