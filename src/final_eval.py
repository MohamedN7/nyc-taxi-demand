"""Évaluation finale sur le test (octobre à décembre 2025) : à lancer UNE SEULE FOIS."""
import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.stats import binom
from torch import nn

from eval_lib import bootstrap_wape, bootstrap_wape_diff, load_dataset, metrics

BASELINE = "mean_same_slot_4w"
FEATURES = ["pu_zone", "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
            "lag_24h", "lag_48h", "lag_168h", "mean_same_hour_7d", "mean_same_hour_28d",
            "prev_day_total", "mean_day_total_7d", "mean_same_slot_4w"]
N_TREES = 3092          # nombre d'arbres trouvé en validation (poisson_sans_meteo)
PARAMS = {"objective": "poisson", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}
RAIN_MM, THRESHOLD = 0.8, 4.0

df = load_dataset()
df["pu_zone"] = df["pu_zone"].astype(int).astype("category")
fit = df[df["split"].isin(["train", "val"])]
test = df[df["split"] == "test"].copy()
print(f"Test : {len(test):,} lignes ({test['date'].min():%d/%m/%Y} au {test['date'].max():%d/%m/%Y})")

# ---------- 1. LightGBM de l'étape 5 (entraîné jusqu'au 30 juin) ----------
lgb_june = lgb.Booster(model_file="models/lgbm_poisson_sans_meteo.txt")
test["pred_lgbm_juin"] = np.clip(lgb_june.predict(test[FEATURES]), 0, None)

# ---------- 2. LightGBM final (réentraîné jusqu'au 30 septembre, mêmes réglages) ----------
final = lgb.train(PARAMS, lgb.Dataset(fit[FEATURES], fit["n_pickups"]), num_boost_round=N_TREES)
final.save_model("models/lgbm_final.txt")
test["pred_lgbm_final"] = np.clip(final.predict(test[FEATURES]), 0, None)

# ---------- 3. LSTM de l'étape 6 (même préparation des données qu'à l'entraînement) ----------
HIST_DAYS = 7
d2 = df.sort_values(["pu_zone", "hour"])
zones = np.sort(d2["pu_zone"].astype(int).unique())
hours = np.sort(d2["hour"].unique())
Z, H = len(zones), len(hours)
D = H // 24
demand = d2["n_pickups"].to_numpy(float).reshape(Z, H)
baseline = d2[BASELINE].to_numpy(float).reshape(Z, H)
first = d2[d2["pu_zone"].astype(int) == zones[0]].iloc[::24]
dow = first["day_of_week"].to_numpy(int) - 1
month = first["month"].to_numpy(float)
day_feat = np.concatenate([np.eye(7)[dow], first[["is_holiday"]].to_numpy(float),
                           np.sin(2 * np.pi * month / 12)[:, None],
                           np.cos(2 * np.pi * month / 12)[:, None]], axis=1)
day_split = first["split"].to_numpy()
train_days = np.where((day_split == "train") & (np.arange(D) >= HIST_DAYS))[0]
test_days = np.where(day_split == "test")[0]
scale = demand[:, (train_days[:, None] * 24 + np.arange(24)).ravel()].mean(axis=1)
hod = np.arange(HIST_DAYS * 24) % 24
hour_enc = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)], axis=1)


class DemandLSTM(nn.Module):          # identique à train_lstm.py
    def __init__(self, n_zones, hidden=64, zone_dim=8, day_dim=10):
        super().__init__()
        self.zone_emb = nn.Embedding(n_zones, zone_dim)
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + zone_dim + day_dim + 24, 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, 24))

    def forward(self, x_hist, base, day, zone):
        _, (h, _) = self.lstm(x_hist)
        return base + self.head(torch.cat([h[-1], self.zone_emb(zone), day, base], dim=1))


model = DemandLSTM(Z)
model.load_state_dict(torch.load("models/lstm.pt"))
model.eval()
z_idx = np.repeat(np.arange(Z), len(test_days))
d_idx = np.tile(test_days, Z)
hist_pos = (d_idx[:, None] - HIST_DAYS) * 24 + np.arange(HIST_DAYS * 24)
tgt_pos = d_idx[:, None] * 24 + np.arange(24)
s = scale[z_idx][:, None]
x_hist = np.concatenate([(demand[z_idx[:, None], hist_pos] / s)[..., None],
                         np.broadcast_to(hour_enc, (len(z_idx), HIST_DAYS * 24, 2))], axis=2)
f = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
with torch.no_grad():
    out = model(f(x_hist), f(baseline[z_idx[:, None], tgt_pos] / s), f(day_feat[d_idx]), torch.tensor(z_idx))
lstm_pred = (out.numpy() * s).clip(min=0)
lstm = pd.DataFrame({"zone_int": np.repeat(zones[z_idx], 24), "hour": hours[tgt_pos.ravel()],
                     "pred_lstm_juin": lstm_pred.ravel()})
test["zone_int"] = test["pu_zone"].astype(int)
test = test.merge(lstm, on=["zone_int", "hour"], how="left")
assert test["pred_lstm_juin"].notna().all(), "prévisions LSTM manquantes"

# ---------- 4. Résultats ----------
models = {"référence : moyenne 4 semaines": BASELINE, "LSTM (jusqu'au 30/06)": "pred_lstm_juin",
          "LightGBM (jusqu'au 30/06)": "pred_lgbm_juin", "LightGBM final (jusqu'au 30/09)": "pred_lgbm_final"}
val_wape = {BASELINE: 0.201, "pred_lstm_juin": 0.151, "pred_lgbm_juin": 0.141, "pred_lgbm_final": None}
rows = []
for name, col in models.items():
    m = metrics(test["n_pickups"], test[col])
    lo, hi = bootstrap_wape(test, col)
    rows.append({"modèle": name, **{k: round(v, 3) for k, v in m.items()},
                 "WAPE IC 95 %": f"[{lo:.3f} ; {hi:.3f}]", "WAPE validation": val_wape[col]})
table = pd.DataFrame(rows)
print("\n=== TEST (octobre à décembre 2025) ===")
print(table.to_string(index=False))
table.to_csv("reports/final_test.csv", index=False)

print()
for a, b in [("pred_lgbm_final", BASELINE), ("pred_lgbm_juin", "pred_lstm_juin"),
             ("pred_lgbm_final", "pred_lgbm_juin")]:
    lo, hi = bootstrap_wape_diff(test, a, b)
    print(f"WAPE({a}) - WAPE({b}) : IC 95 % [{lo:+.3f} ; {hi:+.3f}]")

print("\nWAPE par mois :")
for mth, g in test.groupby(test["date"].dt.month):
    print(f"  mois {mth} : modèle final {metrics(g['n_pickups'], g['pred_lgbm_final'])['WAPE']:.3f}, "
          f"référence {metrics(g['n_pickups'], g[BASELINE])['WAPE']:.3f}")

# ---------- 5. Anomalies sur le test, avec le modèle final ----------
def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    return (x - med) / (1.4826 * np.median(np.abs(x - med)))


test["r"] = (test["n_pickups"] - test["pred_lgbm_final"]) / np.sqrt(test["pred_lgbm_final"] + 1)
test["z"] = test.groupby("zone_int")["r"].transform(robust_z)
city = (test.assign(flag=test["z"].abs() > THRESHOLD).groupby("hour")
            .agg(y=("n_pickups", "sum"), pred=("pred_lgbm_final", "sum"), zones=("flag", "sum")).reset_index())
city["hour"] = pd.to_datetime(city["hour"]).astype("datetime64[ns]")
city["z"] = robust_z((city["y"] - city["pred"]) / np.sqrt(city["pred"] + 1))
a = city[city["z"].abs() > THRESHOLD].copy()
a["sign"] = np.sign(a["z"])
a["ep"] = ((a["hour"].diff() != pd.Timedelta(hours=1)) | (a["sign"] != a["sign"].shift())).cumsum()
episodes = (a.groupby("ep").agg(debut=("hour", "min"), fin=("hour", "max"), heures=("hour", "size"),
                                z_max=("z", lambda v: v.loc[v.abs().idxmax()]), zones=("zones", "max"))
              .assign(force=lambda e: e["z_max"].abs()).sort_values("force", ascending=False))
print(f"\n=== Anomalies du test : {len(a)} heures anormales, {len(episodes)} épisodes ===")
print(episodes.head(10).drop(columns="force").round(1).to_string(index=False))

events = pd.read_csv("config/events_test.csv", parse_dates=["debut", "fin"])
print("\nÉvénements listés avant l'évaluation :")
for _, e in events.iterrows():
    w = city[(city["hour"] >= e["debut"]) & (city["hour"] <= e["fin"])]
    zx = w["z"].loc[w["z"].abs().idxmax()]
    print(f"  {e['evenement']} : {'DÉTECTÉ' if (w['z'].abs() > THRESHOLD).any() else 'non détecté'} "
          f"(score ville extrême {zx:+.1f})")

# ---------- 6. Hypothèse de la pluie, écrite avant l'évaluation ----------
w = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob("data/raw/weather/*.parquet")))
w["time"] = pd.to_datetime(w["time"]).astype("datetime64[ns]")
rain = w[(w["time"] >= "2025-10-01") & (w["time"] <= "2025-12-31 23:00")].set_index("time")["precipitation"]
pos = episodes.head(10)
pos = pos[pos["z_max"] > 0]
hits = sum(rain[(rain.index >= e["debut"] - pd.Timedelta(hours=2)) & (rain.index <= e["fin"])].max() > RAIN_MM
           for _, e in pos.iterrows())
base = np.mean([(rain.rolling(int(e["heures"]) + 2).max() > RAIN_MM).mean() for _, e in pos.iterrows()])
print(f"\nPluie : {hits} épisodes pluvieux sur {len(pos)} au-dessus de la prévision "
      f"(taux de base {base:.0%}) ; probabilité par hasard : {binom.sf(hits - 1, len(pos), base):.4f}")