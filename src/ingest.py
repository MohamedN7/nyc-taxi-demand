"""Étape 1 : télécharge les trajets de taxis jaunes, la table des zones et la météo horaire."""
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

YEARS = [2023, 2024, 2025]
RAW = Path("data/raw")
TRIPS_DIR = RAW / "yellow"
WEATHER_DIR = RAW / "weather"


def download(url, dest):
    """Télécharge url vers dest, sauf si le fichier existe déjà.
    Écrit d'abord dans un fichier .part, renommé à la fin : un téléchargement
    interrompu ne laisse jamais de fichier incomplet sous le nom final."""
    if dest.exists():
        return "déjà présent"
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code in (403, 404):          # fichier pas encore publié
            return "absent"
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):   # par blocs de 1 Mo
                f.write(chunk)
    tmp.rename(dest)
    return "téléchargé"


def download_weather(year):
    """Météo horaire à Central Park, en heure locale de New York."""
    dest = WEATHER_DIR / f"nyc_hourly_{year}.parquet"
    if dest.exists():
        return "déjà présent"
    params = {
        "latitude": 40.7829, "longitude": -73.9654,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m",
        "timezone": "America/New_York",
    }
    r = requests.get(WEATHER_URL, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.to_parquet(dest, index=False)
    return f"téléchargé, {len(df)} heures"


# ---------------- Trajets ----------------
TRIPS_DIR.mkdir(parents=True, exist_ok=True)
schemas = {}          # schéma -> liste des fichiers qui l'ont
total_rows = 0

for year in YEARS:
    for month in range(1, 13):
        name = f"yellow_tripdata_{year}-{month:02d}.parquet"
        dest = TRIPS_DIR / name
        status = download(f"{BASE_URL}/{name}", dest)
        if status == "absent":
            print(f"{name} : pas encore publié")
            continue
        try:
            n_rows = pq.ParquetFile(dest).metadata.num_rows   # relit l'en-tête du fichier
            schema = pq.read_schema(dest)
        except Exception as e:
            print(f"{name} : fichier illisible ({e}), supprimé")
            dest.unlink()
            continue
        key = tuple((f.name, str(f.type)) for f in schema)
        schemas.setdefault(key, []).append(name)
        total_rows += n_rows
        print(f"{name} : {status}, {n_rows:,} trajets")
        if status == "téléchargé":
            time.sleep(1)                        # on ménage le serveur

size_gb = sum(p.stat().st_size for p in TRIPS_DIR.glob("*.parquet")) / 1e9
print(f"\nTotal : {total_rows:,} trajets, {size_gb:.2f} Go")

# ---------------- Contrôle des schémas ----------------
print(f"\n{len(schemas)} schéma(s) différent(s) entre les fichiers")
reference = set(next(iter(schemas)))
for key, files in schemas.items():
    diff = sorted(set(key) ^ reference)
    print(f"- {len(files)} fichier(s), ex. {files[0]} ; différences avec le premier : {diff or 'aucune'}")

# ---------------- Zones et météo ----------------
print("\nZones :", download(ZONES_URL, RAW / "taxi_zone_lookup.csv"))
WEATHER_DIR.mkdir(parents=True, exist_ok=True)
for year in YEARS:
    print(f"Météo {year} :", download_weather(year))