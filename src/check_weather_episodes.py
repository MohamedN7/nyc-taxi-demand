"""Pluie pendant les épisodes anormaux : une explication possible, vérifiée après coup."""
import glob

import pandas as pd

ep = pd.read_csv("reports/anomalies_city_episodes.csv", parse_dates=["debut", "fin"]).head(10)
w = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob("data/raw/weather/*.parquet")))
w["time"] = pd.to_datetime(w["time"])

print("Pluie horaire habituelle en été (95e centile) :",
      f"{w[w['time'].dt.month.isin([7, 8, 9])]['precipitation'].quantile(0.95):.1f} mm")
for _, e in ep.iterrows():
    win = w[(w["time"] >= e["debut"] - pd.Timedelta(hours=2)) & (w["time"] <= e["fin"])]
    print(f"{e['debut']:%d/%m %Hh}-{e['fin']:%Hh} (score {e['z_max']:+.1f}) : "
          f"pluie max {win['precipitation'].max():.1f} mm/h, total {win['precipitation'].sum():.1f} mm, "
          f"{win['temperature_2m'].mean():.0f} °C")