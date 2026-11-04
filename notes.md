# Journal du projet : prévision de la demande de taxis à New York

## Étape 0 : environnement et exploration (janvier 2024)

**Environnement**
- Ubuntu 24.04 sous WSL 2, Python 3.12, Java 21 (OpenJDK), PySpark en mode local (`local[*]`, 16 cœurs)
- pandas maintenu en version < 3 (pas encore complètement pris en charge par PySpark)

**Exploration d'un mois (janvier 2024)**
- 2 964 624 trajets, 19 colonnes
- Contrôles de qualité :
  - distance nulle : 60 371 trajets (environ 2 %), à traiter à l'étape 2 (erreurs ou courses annulées ?)
  - durée de plus de 5 h : 1 814
  - durée nulle ou négative : 870
  - date hors janvier 2024 : 15 (erreurs d'horloge)
- Zones les plus demandées : JFK Airport (145 240), Midtown Center, Upper East Side South et North, LaGuardia en 9e position
- Table zone x heure : 77 547 lignes, contre environ 197 000 possibles (265 zones x 744 heures)
  - les couples (zone, heure) sans aucun trajet sont absents : il faudra les ajouter avec la valeur 0

## Étape 1 : ingestion (2023 à 2025)

**Données téléchargées**
- Trajets : 36 fichiers mensuels, 128 202 548 trajets, 2,16 Go
- Table des zones : `taxi_zone_lookup.csv`
- Météo horaire à Central Park (Open-Meteo) : 8 760 heures en 2023 et 2025, 8 784 en 2024
  - variables : température, précipitations, neige, vent ; en heure locale de New York

**Script `src/ingest.py`**
- idempotent : relancé, il ne retélécharge que les fichiers manquants (vérifié avec janvier 2024)
- écriture dans un fichier `.part`, renommé à la fin : aucun fichier incomplet en cas d'interruption
- vérification de chaque fichier (lecture des métadonnées Parquet) et contrôle des schémas

**Dérive de schéma : 3 schémas différents**
- Groupe A : janvier 2023 seul
- Groupe B : février 2023 à décembre 2024 (23 fichiers)
- Groupe C : 2025 (12 fichiers) : groupe B + nouvelle colonne `cbd_congestion_fee` (péage urbain de Manhattan)
- Différences :
  - types : `VendorID`, `PULocationID`, `DOLocationID` en entier 32 ou 64 bits ; `passenger_count` et `RatecodeID` en entier ou en décimal
  - nom : `airport_fee` ou `Airport_fee`
  - type de texte de `store_and_fwd_flag` (`string` ou `large_string`)
- Décision pour l'étape 2 : lire chaque mois séparément, ne garder que les colonnes utiles, les convertir dans un schéma commun, puis assembler

**Observations à garder en tête**
- Tendance à la hausse : environ 3 M de trajets par mois en 2023, 4 à 4,6 M en 2025 (cause non vérifiée)
  - le modèle devra gérer cette tendance : découpage dans le temps, test sur la fin de 2025
- Changement d'heure : les heures des taxis et de la météo sont en heure locale ; une heure manque ou est doublée chaque année, à traiter lors de la jointure (étape 3)

## Prochaine étape

Étape 2 : nettoyage et agrégation avec Spark (harmonisation des schémas, filtres, table zone x heure complète avec les zéros, rapport de qualité)