"""Prépare les fichiers servis par l'API : modèle, variables des 56 zones, zones et épisodes."""
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from eval_lib import load_dataset

OUT = Path("data/serving")
MODEL = "models/lgbm_poisson_sans_meteo.txt"
OUT.mkdir(parents=True, exist_ok=True)

shutil.copy(MODEL, OUT / "model.txt")
feature_names = lgb.Booster(model_file=MODEL).feature_name()

# Variables des 56 zones, de l'entraînement et de la validation seulement (le test reste verrouillé)
df = load_dataset()
df = df[df["split"].isin(["train", "val"])]
cols = ["pu_zone", "hour", "date", "n_pickups"] + [c for c in feature_names if c != "pu_zone"]
df[list(dict.fromkeys(cols))].to_parquet(OUT / "features.parquet", index=False)

lookup = pd.read_csv("data/raw/taxi_zone_lookup.csv").set_index("LocationID")
zones = json.loads(Path("config/zones.json").read_text())
(OUT / "zones.json").write_text(json.dumps(
    [{"id": z, "name": lookup.loc[z, "Zone"], "borough": lookup.loc[z, "Borough"]} for z in zones],
    ensure_ascii=False, indent=2))

shutil.copy("reports/anomalies_city_episodes.csv", OUT / "anomaly_episodes.csv")

for f in sorted(OUT.iterdir()):
    print(f"{f.name} : {f.stat().st_size / 1e6:.1f} Mo")
print(f"Dates servies : {df['date'].min():%Y-%m-%d} au {df['date'].max():%Y-%m-%d}")