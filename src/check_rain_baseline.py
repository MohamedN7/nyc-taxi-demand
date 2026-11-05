"""Taux de base : part des fenêtres d'été 2025 contenant au moins une heure de pluie > 0,8 mm."""
import glob

import pandas as pd

w = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob("data/raw/weather/*.parquet")))
w["time"] = pd.to_datetime(w["time"]).astype("datetime64[ns]")
summer = (w[(w["time"].dt.year == 2025) & w["time"].dt.month.isin([7, 8, 9])]
          .set_index("time")["precipitation"])
for hours in (3, 5, 8):
    share = (summer.rolling(hours).max() > 0.8).mean()
    print(f"Fenêtres de {hours} h avec au moins une heure de pluie > 0,8 mm : {share:.0%}")