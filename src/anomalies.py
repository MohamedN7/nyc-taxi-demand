"""Étape 7 : détection d'anomalies à partir des écarts entre la demande réelle et la prévision LightGBM."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # pas de fenêtre : on enregistre les figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRED = "data/processed/val_predictions_lgbm.parquet"
EVENTS = "config/events.csv"
THRESHOLD = 4.0
Path("reports/figures").mkdir(parents=True, exist_ok=True)
zone_names = pd.read_csv("data/raw/taxi_zone_lookup.csv").set_index("LocationID")["Zone"]


def robust_z(x):
    """Score robuste : écart à la médiane, en nombre de MAD (x 1,4826 pour être comparable à un écart-type)."""
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * mad)


df = pd.read_parquet(PRED)
df["pu_zone"] = df["pu_zone"].astype(int)
df["hour"] = pd.to_datetime(df["hour"]).astype("datetime64[ns]")

# ---------- 1. Niveau zone x heure ----------
df["resid"] = df["n_pickups"] - df["pred"]
df["r"] = df["resid"] / np.sqrt(df["pred"] + 1)                 # écart rapporté à la taille de l'heure
df["z"] = df.groupby("pu_zone")["r"].transform(robust_z)        # puis à la dispersion habituelle de la zone

print("=== Anomalies zone x heure (validation, 123 648 heures) ===")
for t in (3, 4, 5):
    n = (df["z"].abs() > t).sum()
    print(f"seuil {t} : {n} heures anormales ({100 * n / len(df):.2f} %), "
          f"dont {(df['z'] > t).sum()} au-dessus et {(df['z'] < -t).sum()} en dessous de la prévision")

top = df.loc[df["z"].abs().sort_values(ascending=False).index[:15]].copy()
top["zone"] = top["pu_zone"].map(zone_names)
print("\nLes 15 écarts les plus forts :")
print(top[["hour", "pu_zone", "zone", "n_pickups", "pred", "z"]].round(1).to_string(index=False))
top.to_csv("reports/anomalies_zone_hour_top.csv", index=False)

# ---------- 2. Niveau ville, puis épisodes ----------
city = (df.assign(flag=df["z"].abs() > THRESHOLD)
          .groupby("hour")
          .agg(y=("n_pickups", "sum"), pred=("pred", "sum"), zones_signalees=("flag", "sum"))
          .reset_index())
city["r"] = (city["y"] - city["pred"]) / np.sqrt(city["pred"] + 1)
city["z"] = robust_z(city["r"])
city["anomalous"] = city["z"].abs() > THRESHOLD

a = city[city["anomalous"]].copy()
a["ecart"] = a["y"] - a["pred"]
a["sign"] = np.sign(a["z"])
new_episode = (a["hour"].diff() != pd.Timedelta(hours=1)) | (a["sign"] != a["sign"].shift())
a["episode"] = new_episode.cumsum()
episodes = (a.groupby("episode")
              .agg(debut=("hour", "min"), fin=("hour", "max"), heures=("hour", "size"),
                   z_max=("z", lambda s: s.loc[s.abs().idxmax()]),
                   ecart_trajets=("ecart", "sum"), zones_signalees=("zones_signalees", "max"))
              .assign(force=lambda e: e["z_max"].abs())
              .sort_values("force", ascending=False)
              .drop(columns="force"))
episodes["jour"] = episodes["debut"].dt.day_name()
print(f"\n=== Ville entière : {city['anomalous'].sum()} heures anormales, {len(episodes)} épisodes ===")
print(episodes.head(10).round(1).to_string(index=False))
episodes.to_csv("reports/anomalies_city_episodes.csv", index=False)

# ---------- 3. Confrontation avec les événements listés à l'avance ----------
events = pd.read_csv(EVENTS, parse_dates=["debut", "fin"])
print("\n=== Événements connus (listés avant l'analyse) ===")
for _, e in events.iterrows():
    w = city[(city["hour"] >= e["debut"]) & (city["hour"] <= e["fin"])]
    z_extreme = w["z"].loc[w["z"].abs().idxmax()]
    print(f"{e['evenement']} : {'DÉTECTÉ' if w['anomalous'].any() else 'non détecté'} "
          f"(score ville extrême {z_extreme:+.1f}, jusqu'à {int(w['zones_signalees'].max())} zones signalées)")


def matches_event(row):
    hit = events[(events["debut"] <= row["fin"]) & (events["fin"] >= row["debut"])]
    return ", ".join(hit["evenement"]) if len(hit) else "-"


episodes["evenement_connu"] = episodes.apply(matches_event, axis=1)
print("\nLes 10 épisodes les plus forts correspondent-ils à un événement connu ?")
print(episodes.head(10)[["debut", "fin", "jour", "z_max", "evenement_connu"]].round(1).to_string(index=False))

# ---------- 4. Figure autour de l'épisode le plus fort ----------
start = episodes.iloc[0]["debut"]
w = city[(city["hour"] >= start - pd.Timedelta(days=2)) & (city["hour"] <= start + pd.Timedelta(days=2))]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(w["hour"], w["y"], label="demande réelle")
ax.plot(w["hour"], w["pred"], label="prévision LightGBM", linestyle="--")
ax.scatter(w.loc[w["anomalous"], "hour"], w.loc[w["anomalous"], "y"], color="red", zorder=3,
           label=f"heure anormale (|score| > {THRESHOLD:g})")
ax.set_title("Prises en charge par heure, 56 zones, autour de l'épisode le plus fort")
ax.set_ylabel("prises en charge par heure")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("reports/figures/episode_le_plus_fort.png", dpi=150)
print("\nFigure : reports/figures/episode_le_plus_fort.png")