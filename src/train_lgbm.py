"""Étape 5 : LightGBM, comparé à la meilleure référence, avec suivi des essais dans MLflow."""
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from eval_lib import bootstrap_wape, bootstrap_wape_diff, load_dataset, metrics

BASELINE = "mean_same_slot_4w"           # la référence à battre (WAPE 20,1 % en validation)

ZONE = ["pu_zone"]
CALENDAR = ["hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday"]
HISTORY = ["lag_24h", "lag_48h", "lag_168h", "mean_same_hour_7d", "mean_same_hour_28d",
           "prev_day_total", "mean_day_total_7d", "mean_same_slot_4w"]
WEATHER = ["temp_c", "precip_mm", "snow_cm", "wind_kmh"]
ALL = ZONE + CALENDAR + HISTORY + WEATHER

CONFIGS = [
    {"run": "l2_toutes_variables",      "objective": "regression",    "features": ALL},
    {"run": "l1_toutes_variables",      "objective": "regression_l1", "features": ALL},
    {"run": "poisson_toutes_variables", "objective": "poisson",       "features": ALL},
    {"run": "poisson_sans_meteo",       "objective": "poisson",       "features": ZONE + CALENDAR + HISTORY},
]
PARAMS = {"learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
          "metric": "l1", "verbose": -1, "seed": 42}

# ---------- Données ----------
df = load_dataset()
df["pu_zone"] = df["pu_zone"].astype(int).astype("category")   # la zone est une catégorie
train = df[df["split"] == "train"]
val = df[df["split"] == "val"].copy()
print(f"Entraînement : {len(train):,} lignes ; validation : {len(val):,} lignes")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("lightgbm")
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

results = [{"run": "référence : moyenne 4 semaines",
            **metrics(val["n_pickups"], val[BASELINE]), "écart avec la référence (IC 95 %)": "-"}]

for cfg in CONFIGS:
    feats, col = cfg["features"], f"pred_{cfg['run']}"
    params = {**PARAMS, "objective": cfg["objective"]}
    dtrain = lgb.Dataset(train[feats], train["n_pickups"])
    dval = lgb.Dataset(val[feats], val["n_pickups"], reference=dtrain)

    t0 = time.time()
    with mlflow.start_run(run_name=cfg["run"]):
        booster = lgb.train(params, dtrain, num_boost_round=20000, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        seconds = time.time() - t0
        val[col] = np.clip(booster.predict(val[feats], num_iteration=booster.best_iteration), 0, None)

        m = metrics(val["n_pickups"], val[col])
        lo, hi = bootstrap_wape(val, col)
        dlo, dhi = bootstrap_wape_diff(val, col, BASELINE)

        mlflow.log_params({**params, "n_features": len(feats), "features": ",".join(feats),
                           "best_iteration": booster.best_iteration})
        mlflow.log_metrics({**{f"val_{k}": v for k, v in m.items()},
                            "val_WAPE_ci_low": lo, "val_WAPE_ci_high": hi,
                            "diff_vs_baseline_low": dlo, "diff_vs_baseline_high": dhi,
                            "train_seconds": seconds})

        imp = (pd.DataFrame({"variable": feats, "gain": booster.feature_importance("gain")})
               .assign(part=lambda x: x["gain"] / x["gain"].sum())
               .sort_values("gain", ascending=False))
        imp_path = f"reports/importance_{cfg['run']}.csv"
        imp.to_csv(imp_path, index=False)
        model_path = f"models/lgbm_{cfg['run']}.txt"
        booster.save_model(model_path)
        mlflow.log_artifact(imp_path)
        mlflow.log_artifact(model_path)

    results.append({"run": cfg["run"], **m,
                    "écart avec la référence (IC 95 %)": f"[{dlo:+.3f} ; {dhi:+.3f}]"})
    print(f"{cfg['run']} : WAPE {m['WAPE']:.3f} [{lo:.3f} ; {hi:.3f}], "
          f"{booster.best_iteration} arbres, {seconds:.0f} s")

# ---------- Bilan ----------
table = pd.DataFrame(results).round(3)
print("\n=== Validation (juillet à septembre 2025) ===")
print(table.to_string(index=False))
table.to_csv("reports/lgbm_val.csv", index=False)

wlo, whi = bootstrap_wape_diff(val, "pred_poisson_sans_meteo", "pred_poisson_toutes_variables")
print(f"\nApport de la météo : WAPE(sans) - WAPE(avec) : IC 95 % [{wlo:+.3f} ; {whi:+.3f}]")

best = min(results[1:], key=lambda r: r["WAPE"])["run"]
best_col = f"pred_{best}"
print(f"\nMeilleur modèle : {best}")
print("Importance des variables (10 premières) :")
print(pd.read_csv(f"reports/importance_{best}.csv").head(10)[["variable", "part"]]
      .round(3).to_string(index=False))

per_zone = val.groupby("pu_zone", observed=True).apply(
    lambda g: pd.Series({"modele": metrics(g["n_pickups"], g[best_col])["WAPE"],
                         "reference": metrics(g["n_pickups"], g[BASELINE])["WAPE"]}),
    include_groups=False)
print(f"\nZones où le modèle bat la référence : {(per_zone['modele'] < per_zone['reference']).sum()} / {len(per_zone)}")

# Prévisions de validation du meilleur modèle, pour la détection d'anomalies (étape 7)
(val[["pu_zone", "hour", "date", "n_pickups", BASELINE, best_col]]
    .rename(columns={best_col: "pred"})
    .to_parquet("data/processed/val_predictions_lgbm.parquet", index=False))