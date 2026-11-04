"""Protocole d'évaluation commun à tous les modèles (étapes 4 à 6)."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

FEATURES = "data/processed/features.parquet"
ZONES_FILE = Path("config/zones.json")
N_ZONES = 56
SPLITS = {
    "train": ("2023-01-08", "2025-06-30"),
    "val":   ("2025-07-01", "2025-09-30"),
    "test":  ("2025-10-01", "2025-12-31"),
}


def select_zones():
    """Les N_ZONES zones les plus demandées, classées sur l'ENTRAÎNEMENT seulement."""
    start, end = SPLITS["train"]
    zones = duckdb.sql(f"""
        SELECT pu_zone, SUM(n_pickups) AS n
        FROM '{FEATURES}'
        WHERE date BETWEEN DATE '{start}' AND DATE '{end}'
        GROUP BY pu_zone ORDER BY n DESC LIMIT {N_ZONES}
    """).df()["pu_zone"].astype(int).tolist()
    ZONES_FILE.parent.mkdir(exist_ok=True)
    ZONES_FILE.write_text(json.dumps(zones))
    return zones


def load_dataset():
    """Variables des zones retenues, plus la référence 'moyenne des 4 dernières semaines'."""
    zones = json.loads(ZONES_FILE.read_text())
    df = duckdb.sql(f"""
        SELECT *,
               AVG(n_pickups) OVER (PARTITION BY pu_zone, hour_of_day, day_of_week ORDER BY hour
                                    ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS mean_same_slot_4w
        FROM '{FEATURES}'
        WHERE pu_zone IN ({",".join(map(str, zones))})
        ORDER BY pu_zone, hour
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    # Types pandas « à valeurs manquantes » (Int64, Float64, boolean) -> float64 NumPy
    for c in df.columns:
        if pd.api.types.is_extension_array_dtype(df[c]) and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float64")
    df["split"] = "excluded"
    for name, (start, end) in SPLITS.items():
        df.loc[df["date"].between(pd.Timestamp(start), pd.Timestamp(end)), "split"] = name
    return df


def metrics(y, pred):
    """MAE, RMSE, WAPE et biais (WAPE et biais en proportion de la demande réelle)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "WAPE": float(np.sum(np.abs(err)) / np.sum(y)),
        "bias": float(np.sum(err) / np.sum(y)),
    }


def _daily_errors(df, pred_cols):
    """Somme des erreurs absolues et de la demande, par jour : l'unité du bootstrap."""
    out = pd.DataFrame({"y": df["n_pickups"].values, "date": df["date"].values})
    for c in pred_cols:
        out[c] = np.abs(df[c].values - df["n_pickups"].values)
    return out.groupby("date").sum()


def bootstrap_wape(df, pred, n_boot=1000, seed=0):
    """Intervalle de confiance à 95 % de la WAPE, en tirant des jours au hasard."""
    g = _daily_errors(df, [pred])
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    wape = g[pred].values[idx].sum(1) / g["y"].values[idx].sum(1)
    return tuple(np.percentile(wape, [2.5, 97.5]))


def bootstrap_wape_diff(df, pred_a, pred_b, n_boot=1000, seed=0):
    """IC à 95 % de WAPE(a) - WAPE(b). Si l'intervalle ne contient pas 0, l'écart est réel."""
    g = _daily_errors(df, [pred_a, pred_b])
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    y = g["y"].values[idx].sum(1)
    diff = g[pred_a].values[idx].sum(1) / y - g[pred_b].values[idx].sum(1) / y
    return tuple(np.percentile(diff, [2.5, 97.5]))