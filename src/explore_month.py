"""Premier contact avec Spark : un mois de trajets de taxis jaunes."""
from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder
         .appName("explore-month")
         .master("local[*]")                    # tous les cœurs du PC
         .config("spark.driver.memory", "4g")
         .getOrCreate())

# --- Lecture (transformation : rien n'est encore calculé) ---
trips = spark.read.parquet("data/raw/yellow/yellow_tripdata_2024-01.parquet")
trips.printSchema()

# --- Action : Spark lit vraiment le fichier ---
print("Nombre de trajets :", trips.count())
trips.select("tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID",
             "DOLocationID", "trip_distance", "total_amount").show(5, truncate=False)

# --- Nouvelle colonne : durée du trajet en minutes ---
trips = trips.withColumn(
    "duration_min",
    (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60,
)

# --- Contrôles de qualité : combien de trajets aberrants ? ---
trips.select(
    F.sum((F.col("duration_min") <= 0).cast("int")).alias("duree_nulle_ou_negative"),
    F.sum((F.col("duration_min") > 300).cast("int")).alias("duree_plus_de_5h"),
    F.sum((F.col("trip_distance") <= 0).cast("int")).alias("distance_nulle"),
    F.sum((F.month("tpep_pickup_datetime") != 1).cast("int")).alias("hors_janvier_2024"),
).show()

# --- Jointure avec les zones : les 10 zones les plus demandées ---
zones = spark.read.csv("data/raw/taxi_zone_lookup.csv", header=True, inferSchema=True)
(trips.groupBy("PULocationID").count()
      .join(zones, F.col("PULocationID") == F.col("LocationID"), "left")
      .select("PULocationID", "Borough", "Zone", "count")
      .orderBy(F.desc("count"))
      .show(10, truncate=False))

# --- Agrégation zone x heure : la forme des données du futur modèle ---
hourly = (trips
          .withColumn("pickup_hour", F.date_trunc("hour", "tpep_pickup_datetime"))
          .groupBy("PULocationID", "pickup_hour")
          .count())
print("Nombre de lignes zone x heure :", hourly.count())
hourly.orderBy("PULocationID", "pickup_hour").show(5)

input("Spark UI : http://localhost:4040 ; appuie sur Entrée pour quitter...")
spark.stop()