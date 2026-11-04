"""Étape 4 : zones retenues, découpage dans le temps et modèles de référence (sur la validation)."""
import duckdb
import pandas as pd

from eval_lib import (FEATURES, SPLITS, bootstrap_wape, bootstrap_wape_diff,
                      load_dataset, metrics, select_zones)

zones = select_zones()
start, end = SPLITS["train"]
share = duckdb.execute(f"""
    SELECT SUM(CASE WHEN pu_zone IN ({",".join(map(str, zones))}) THEN n_pickups END) / SUM(n_pickups)
    FROM '{FEATURES}' WHERE date BETWEEN DATE '{start}' AND DATE '{end}'
""").fetchall()[0][0]
print(f"{len(zones)} zones retenues, {share:.1%} des prises en charge de l'entraînement")

df = load_dataset()
print("\nLignes par période :")
print(df["split"].value_counts().to_string())

val = df[df["split"] == "val"]
baselines = {
    "veille (lag_24h)": "lag_24h",
    "semaine précédente (lag_168h)": "lag_168h",
    "moyenne 4 semaines, même créneau": "mean_same_slot_4w",
}

rows = []
for name, col in baselines.items():
    m = metrics(val["n_pickups"], val[col])
    lo, hi = bootstrap_wape(val, col)
    rows.append({"modèle": name, **{k: round(v, 3) for k, v in m.items()},
                 "WAPE IC 95 %": f"[{lo:.3f} ; {hi:.3f}]"})
results = pd.DataFrame(rows)
print("\n=== Modèles de référence, période de validation ===")
print(results.to_string(index=False))
results.to_csv("reports/baselines_val.csv", index=False)

best = results.sort_values("WAPE").iloc[0]["modèle"]
best_col = baselines[best]
for name, col in baselines.items():
    if col != best_col:
        lo, hi = bootstrap_wape_diff(val, col, best_col)
        print(f"WAPE({name}) - WAPE({best}) : IC 95 % [{lo:+.3f} ; {hi:+.3f}]")

# Les zones les plus difficiles pour la meilleure référence
zone_names = pd.read_csv("data/raw/taxi_zone_lookup.csv").set_index("LocationID")["Zone"]
per_zone = (val.groupby("pu_zone")
               .apply(lambda g: metrics(g["n_pickups"], g[best_col])["WAPE"], include_groups=False)
               .sort_values(ascending=False))
print(f"\nWAPE médiane par zone ({best}) : {per_zone.median():.3f}")
print("Les 5 zones les plus difficiles :")
for z, w in per_zone.head(5).items():
    print(f"  {z} {zone_names.get(z, '?')} : {w:.3f}")