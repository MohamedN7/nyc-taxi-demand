"""Étape 2 : nettoyage des trajets et table zone x heure complète (avec Spark)."""
from functools import reduce
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession, functions as F

RAW = Path("data/raw/yellow")
ZONES_CSV = "data/raw/taxi_zone_lookup.csv"
OUT = "data/processed/zone_hour"
REPORT = "data/processed/quality_report.csv"
START, END = "2023-01-01 00:00:00", "2025-12-31 23:00:00"
MAX_DURATION_MIN = 300      # 5 heures
MAX_DISTANCE_MI = 100

spark = (SparkSession.builder
         .appName("clean-aggregate")
         .master("local[*]")
         .config("spark.driver.memory", "5g")
         .config("spark.sql.shuffle.partitions", "64")   # nombre de morceaux après un regroupement
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")                  # moins de messages à l'écran


# ---------- 1. Lire chaque mois et l'harmoniser dans un schéma commun ----------
def read_month(path):
    year, month = map(int, path.stem.split("_")[-1].split("-"))   # "yellow_tripdata_2023-01"
    return (spark.read.parquet(str(path))
            .select(
                F.col("tpep_pickup_datetime").alias("pickup_ts"),
                F.col("tpep_dropoff_datetime").alias("dropoff_ts"),
                F.col("PULocationID").cast("int").alias("pu_zone"),
                F.col("DOLocationID").cast("int").alias("do_zone"),
                F.col("trip_distance").cast("double").alias("trip_distance"),
            )
            .withColumn("file_year", F.lit(year))
            .withColumn("file_month", F.lit(month)))


files = sorted(RAW.glob("yellow_tripdata_*.parquet"))
trips = reduce(lambda a, b: a.unionByName(b), [read_month(p) for p in files])
print(f"{len(files)} fichiers lus et harmonisés")

# ---------- 2. Zones valides (on exclut les zones inconnues ou hors de New York) ----------
zones = spark.read.csv(ZONES_CSV, header=True, inferSchema=True)
invalid = zones.filter(F.col("Borough").isNull() | F.col("Borough").isin("Unknown", "N/A"))
print("Zones exclues :", [(r.LocationID, r.Borough, r.Zone) for r in invalid.collect()])
valid_zones = [r.LocationID for r in zones.join(invalid, "LocationID", "left_anti").collect()]
print(f"{len(valid_zones)} zones valides")


# ---------- 3. Marquer les trajets aberrants (sans encore les retirer) ----------
def flag(condition):
    """Une valeur manquante compte comme aberrante."""
    return F.coalesce(condition, F.lit(True))


duration_min = (F.unix_timestamp("dropoff_ts") - F.unix_timestamp("pickup_ts")) / 60
flagged = (trips
    .withColumn("bad_month", flag((F.year("pickup_ts") != F.col("file_year"))
                                  | (F.month("pickup_ts") != F.col("file_month"))))
    .withColumn("bad_duration", flag((duration_min <= 0) | (duration_min > MAX_DURATION_MIN)))
    .withColumn("bad_distance", flag((F.col("trip_distance") < 0)
                                     | (F.col("trip_distance") > MAX_DISTANCE_MI)))
    .withColumn("bad_zone", flag(~F.col("pu_zone").isin(valid_zones)))
    .withColumn("zero_distance", F.coalesce(F.col("trip_distance") == 0, F.lit(False)))
    .withColumn("keep", ~(F.col("bad_month") | F.col("bad_duration")
                          | F.col("bad_distance") | F.col("bad_zone"))))

# ---------- 4. Rapport de qualité, mois par mois ----------
def n(col):
    return F.sum(F.col(col).cast("int"))


report = (flagged.groupBy("file_year", "file_month")
          .agg(F.count("*").alias("total"),
               n("bad_month").alias("bad_month"),
               n("bad_duration").alias("bad_duration"),
               n("bad_distance").alias("bad_distance"),
               n("bad_zone").alias("bad_zone"),
               F.sum((F.col("zero_distance") & F.col("keep")).cast("int")).alias("zero_distance_kept"),
               n("keep").alias("kept"))
          .orderBy("file_year", "file_month")
          .toPandas())
report["removed_pct"] = (100 * (1 - report["kept"] / report["total"])).round(2)
Path(REPORT).parent.mkdir(parents=True, exist_ok=True)
report.to_csv(REPORT, index=False)

tot = report[["total", "bad_month", "bad_duration", "bad_distance", "bad_zone",
              "zero_distance_kept", "kept"]].sum()
print("\n=== Rapport de qualité (3 ans) ===")
print(tot.to_string())
print(f"Part retirée : {100 * (1 - tot['kept'] / tot['total']):.2f} %")

# ---------- 5. Comptage par zone et par heure ----------
counts = (flagged.filter("keep")
          .withColumn("hour_key", F.date_format("pickup_ts", "yyyy-MM-dd HH:00:00"))
          .groupBy("pu_zone", "hour_key")
          .agg(F.count("*").alias("n_pickups")))

# ---------- 6. Grille complète : toutes les zones x toutes les heures ----------
hour_keys = pd.date_range(START, END, freq="h").strftime("%Y-%m-%d %H:00:00").tolist()
hours = spark.createDataFrame([(h,) for h in hour_keys], ["hour_key"])
zone_df = spark.createDataFrame([(z,) for z in valid_zones], ["pu_zone"])
grid = zone_df.crossJoin(hours)

table = (grid.join(counts, ["pu_zone", "hour_key"], "left")   # garde les heures sans trajet
         .fillna({"n_pickups": 0})                             # ... et leur met 0
         .withColumn("hour", F.col("hour_key").cast("timestamp_ntz"))
         .withColumn("year", F.year("hour"))
         .withColumn("month", F.month("hour"))
         .drop("hour_key"))

# overwrite : relancer le script remplace le résultat, sans doublons (idempotence)
table.write.mode("overwrite").partitionBy("year", "month").parquet(OUT)
print(f"\nTable écrite dans {OUT}")

# ---------- 7. Vérifications sur le résultat relu ----------
check = spark.read.parquet(OUT)
n_rows = check.count()
expected = len(valid_zones) * len(hour_keys)
total_pickups = check.agg(F.sum("n_pickups")).first()[0]
n_dup = check.groupBy("pu_zone", "hour").count().filter("count > 1").count()
n_null = check.filter(F.col("n_pickups").isNull() | F.col("hour").isNull()).count()
zero_share = check.filter("n_pickups = 0").count() / n_rows


def ok(cond):
    return "OK" if cond else "ÉCHEC"


print("\n=== Vérifications ===")
print(f"Lignes : {n_rows:,} (attendu {expected:,}) : {ok(n_rows == expected)}")
print(f"Somme des prises en charge : {total_pickups:,} (trajets gardés {int(tot['kept']):,}) : "
      f"{ok(total_pickups == tot['kept'])}")
print(f"Doublons (zone, heure) : {n_dup} : {ok(n_dup == 0)}")
print(f"Valeurs manquantes : {n_null} : {ok(n_null == 0)}")
print(f"Part des heures à 0 trajet : {zero_share:.1%}")

# ---------- 8. Concentration de la demande (pour choisir les zones du modèle) ----------
by_zone = (check.groupBy("pu_zone").agg(F.sum("n_pickups").alias("n")).toPandas()
           .sort_values("n", ascending=False))
cum = by_zone["n"].cumsum() / by_zone["n"].sum()
for p in (0.8, 0.9, 0.95):
    print(f"{int((cum < p).sum()) + 1} zones font {p:.0%} des prises en charge")

spark.stop()