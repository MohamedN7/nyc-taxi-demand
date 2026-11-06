"""Crée un jeu de fichiers « servis » synthétiques, pour tester l'API sans les vraies données (CI)."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = ["pu_zone", "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
            "lag_24h", "lag_48h", "lag_168h", "mean_same_hour_7d", "mean_same_hour_28d",
            "prev_day_total", "mean_day_total_7d", "mean_same_slot_4w"]

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/serving")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

zones = json.loads(Path("config/zones.json").read_text())
hours = pd.date_range("2025-07-01", "2025-07-31 23:00", freq="h")
df = pd.DataFrame([(z, h) for z in zones for h in hours], columns=["pu_zone", "hour"])
df["date"] = df["hour"].dt.normalize()
df["hour_of_day"] = df["hour"].dt.hour.astype(float)
df["day_of_week"] = (df["hour"].dt.dayofweek + 1).astype(float)
df["month"] = df["hour"].dt.month.astype(float)
df["is_weekend"] = (df["day_of_week"] >= 6).astype(float)
df["is_holiday"] = 0.0

# Une demande synthétique : un niveau par zone, et un cycle sur la journée
level = np.repeat(rng.uniform(5, 150, len(zones)), len(hours))
lam = level * (1 + 0.5 * np.sin(2 * np.pi * df["hour_of_day"] / 24))
df["n_pickups"] = rng.poisson(lam).astype(float)
for col in FEATURES[6:]:
    df[col] = lam * rng.uniform(0.8, 1.2, len(df))

X = df[FEATURES].copy()
X["pu_zone"] = X["pu_zone"].astype("category")
booster = lgb.train({"objective": "poisson", "num_leaves": 15, "verbose": -1},
                    lgb.Dataset(X, df["n_pickups"]), num_boost_round=20)
booster.save_model(str(out / "model.txt"))

df[["pu_zone", "hour", "date", "n_pickups"] + FEATURES[1:]].to_parquet(out / "features.parquet", index=False)
(out / "zones.json").write_text(json.dumps([{"id": z, "name": f"Zone {z}", "borough": "Test"} for z in zones]))
pd.DataFrame({"debut": ["2025-07-14 17:00"], "fin": ["2025-07-14 21:00"], "heures": [5], "z_max": [14.2],
              "ecart_trajets": [17000.0], "zones_signalees": [42], "jour": ["Monday"]}
             ).to_csv(out / "anomaly_episodes.csv", index=False)
print(f"Fichiers de test créés dans {out}")