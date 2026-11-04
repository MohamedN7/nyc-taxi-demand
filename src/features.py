"""Étape 3 : variables du modèle calculées en SQL avec DuckDB."""
import duckdb
import holidays
import pandas as pd

con = duckdb.connect("data/processed/taxi.duckdb")   # la base tient dans ce fichier

# ---------- 1. Charger la table zone x heure (directement depuis les Parquet) ----------
con.execute("""
CREATE OR REPLACE TABLE zone_hour AS
SELECT pu_zone, CAST(hour AS TIMESTAMP) AS hour, n_pickups
FROM read_parquet('data/processed/zone_hour/**/*.parquet', hive_partitioning = true)
""")

# ---------- 2. Météo : une ligne par heure locale ----------
# Au changement d'heure d'automne, une heure locale apparaît deux fois : on fait la moyenne.
con.execute("""
CREATE OR REPLACE TABLE weather AS
SELECT CAST(time AS TIMESTAMP) AS hour,
       AVG(temperature_2m) AS temp_c,
       AVG(precipitation)  AS precip_mm,
       AVG(snowfall)       AS snow_cm,
       AVG(wind_speed_10m) AS wind_kmh,
       COUNT(*)            AS n_obs
FROM read_parquet('data/raw/weather/*.parquet')
GROUP BY 1
""")

# ---------- 3. Jours fériés de l'État de New York ----------
hol = holidays.US(state="NY", years=[2023, 2024, 2025])
hol_df = pd.DataFrame({"date": pd.to_datetime(list(hol.keys())), "holiday_name": list(hol.values())})
con.register("hol_df", hol_df)
con.execute("CREATE OR REPLACE TABLE holidays AS SELECT CAST(date AS DATE) AS date, holiday_name FROM hol_df")

# ---------- 4. Variables journalières : total de la veille et moyenne des 7 jours précédents ----------
con.execute("""
CREATE OR REPLACE TABLE daily_feat AS
WITH daily AS (
    SELECT pu_zone, CAST(hour AS DATE) AS date, SUM(n_pickups) AS day_total
    FROM zone_hour
    GROUP BY 1, 2
)
SELECT pu_zone, date,
       LAG(day_total, 1) OVER (PARTITION BY pu_zone ORDER BY date) AS prev_day_total,
       AVG(day_total) OVER (PARTITION BY pu_zone ORDER BY date
                            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS mean_day_total_7d
FROM daily
""")

# ---------- 5. Table finale des variables ----------
con.execute("""
CREATE OR REPLACE TABLE features AS
WITH base AS (
    SELECT pu_zone, hour, n_pickups,
           CAST(hour AS DATE)        AS date,
           EXTRACT(hour FROM hour)   AS hour_of_day,
           ISODOW(hour)              AS day_of_week,      -- 1 = lundi ... 7 = dimanche
           EXTRACT(month FROM hour)  AS month,
           -- demande passée : uniquement des heures connues la veille au soir
           LAG(n_pickups, 24)  OVER (PARTITION BY pu_zone ORDER BY hour) AS lag_24h,
           LAG(n_pickups, 48)  OVER (PARTITION BY pu_zone ORDER BY hour) AS lag_48h,
           LAG(n_pickups, 168) OVER (PARTITION BY pu_zone ORDER BY hour) AS lag_168h,
           -- même heure de la journée, sur les jours précédents
           AVG(n_pickups) OVER (PARTITION BY pu_zone, EXTRACT(hour FROM hour) ORDER BY hour
                                ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)  AS mean_same_hour_7d,
           AVG(n_pickups) OVER (PARTITION BY pu_zone, EXTRACT(hour FROM hour) ORDER BY hour
                                ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS mean_same_hour_28d
    FROM zone_hour
)
SELECT b.*,
       d.prev_day_total,
       d.mean_day_total_7d,
       b.day_of_week >= 6   AS is_weekend,
       h.date IS NOT NULL   AS is_holiday,
       w.temp_c, w.precip_mm, w.snow_cm, w.wind_kmh
FROM base b
LEFT JOIN daily_feat d USING (pu_zone, date)
LEFT JOIN holidays h   ON h.date = b.date
LEFT JOIN weather w    ON w.hour = b.hour
ORDER BY b.pu_zone, b.hour
""")

# ---------- 6. Vérifications ----------
n_base = con.execute("SELECT COUNT(*) FROM zone_hour").fetchone()[0]
n_feat = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
print(f"Lignes : {n_feat:,} (table zone x heure : {n_base:,}) : {'OK' if n_feat == n_base else 'ÉCHEC'}")

# Le décalage de 24 h est-il vraiment la valeur de la veille, à la même heure ?
n_wrong = con.execute("""
SELECT COUNT(*) FROM features f
JOIN zone_hour z ON z.pu_zone = f.pu_zone AND z.hour = f.hour - INTERVAL 24 HOUR
WHERE f.lag_24h <> z.n_pickups
""").fetchone()[0]
print(f"Décalages de 24 h incorrects : {n_wrong} : {'OK' if n_wrong == 0 else 'ÉCHEC'}")

print("\nValeurs manquantes par variable :")
cols = ["lag_24h", "lag_48h", "lag_168h", "mean_same_hour_7d", "mean_same_hour_28d",
        "prev_day_total", "mean_day_total_7d", "temp_c", "precip_mm"]
missing = con.execute("SELECT " + ", ".join(f"COUNT(*) - COUNT({c}) AS {c}" for c in cols)
                      + " FROM features").df()
print(missing.T.rename(columns={0: "manquantes"}).to_string())

print("\nHeures sans météo :", con.execute(
    "SELECT DISTINCT hour FROM features WHERE temp_c IS NULL ORDER BY hour").df()["hour"].tolist())
print("Heures météo en double (fusionnées) :",
      con.execute("SELECT COUNT(*) FROM weather WHERE n_obs > 1").fetchone()[0])
print("Jours fériés :", con.execute("SELECT COUNT(*) FROM holidays").fetchone()[0])

print("\nCorrélation de chaque variable avec la demande :")
print(con.execute("""
SELECT ROUND(CORR(n_pickups, lag_24h), 3)            AS lag_24h,
       ROUND(CORR(n_pickups, lag_168h), 3)           AS lag_168h,
       ROUND(CORR(n_pickups, mean_same_hour_7d), 3)  AS mean_same_hour_7d,
       ROUND(CORR(n_pickups, temp_c), 3)             AS temp_c
FROM features
""").df().to_string(index=False))

# ---------- 7. Export ----------
con.execute("COPY features TO 'data/processed/features.parquet' (FORMAT PARQUET)")
print("\nVariables exportées dans data/processed/features.parquet")
con.close()