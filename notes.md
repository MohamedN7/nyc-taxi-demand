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

## Étape 2 : nettoyage et agrégation avec Spark

**Règles de nettoyage** (sur 128 202 548 trajets)
- durée <= 0 ou > 5 h : 646 466 retirés (0,50 %)
- zone de départ 264 (Unknown) ou 265 (Outside of NYC) : 636 461 retirés (0,50 %)
- distance < 0 ou > 100 miles : 5 787 retirés
- date hors du mois du fichier : 1 364 retirés
- total retiré : 1,00 % ; trajets gardés : 126 918 348
- distance nulle : 2 827 471 trajets gardés (2,2 %), car on prévoit des prises en charge

**Table zone x heure** : `data/processed/zone_hour/` (Parquet, partitionné par année et mois)
- 263 zones x 26 304 heures = 6 917 952 lignes
- vérifications : nombre de lignes, somme des prises en charge = trajets gardés, aucun doublon, aucune valeur manquante
- 51,9 % des heures à 0 trajet
- concentration : 32 zones = 80 %, 43 = 90 %, 56 = 95 % des prises en charge
- décision : modéliser les 56 premières zones, classées sur la période d'entraînement seulement (pas de fuite de données)
- changements d'heure : 2 h n'existe pas au printemps (0 trajet), 1 h compte double en automne

## Étape 3 : variables en SQL (DuckDB)

**Règle** : une variable n'est autorisée que si elle est connue la veille à minuit
(prévision des 24 heures du lendemain). Pas de décalage de moins de 24 h.

**Variables** : calendrier (heure, jour de la semaine, mois, week-end, 45 jours fériés NY),
décalages 24 h, 48 h, 168 h, moyennes de la même heure sur 7 et 28 jours,
total et moyenne journalière de la veille et des 7 jours précédents,
météo (température, précipitations, neige, vent)

**Vérifications** : 6 917 952 lignes, 0 décalage de 24 h incorrect
- valeurs manquantes = 263 x nombre d'heures du début de période : on retirera la 1re semaine de 2023
- météo : une valeur par heure de l'horloge, aucune manquante ni en double

**Corrélations globales** : lag_168h 0,956 ; moyenne 7 j 0,938 ; lag_24h 0,929 ; température 0,032
- gonflées par la taille des zones (calculées sur toutes les zones mélangées)
- météo : effet probablement non linéaire, à mesurer avec et sans à l'étape 5
- hypothèse optimiste : on utilise la météo observée comme une prévision parfaite

## Étape 4 : protocole d'évaluation

**Découpage** : entraînement 08/01/2023 au 30/06/2025 (1 216 320 lignes) ; validation 07 à 09/2025
(123 648) ; test 10 à 12/2025 (123 648), verrouillé jusqu'à la fin ; 1re semaine exclue (9 408)

**Zones** : 56 zones classées sur l'entraînement seulement (`config/zones.json`), 96,0 % des prises en charge

**Mesure principale** : WAPE (pas la MAPE, impossible quand la demande vaut 0) ;
intervalles de confiance par bootstrap sur les jours (pas les heures, corrélées entre elles)

**Références sur la validation**
- veille : WAPE 27,3 % [25,2 ; 29,5]
- semaine précédente : 21,8 % [20,0 ; 23,7]
- moyenne 4 semaines même créneau : 20,1 % [18,5 ; 22,1] ← barre à battre
- écarts significatifs ; biais quasi nul (-0,5 %), cohérent avec la hausse de 2025

**Zones difficiles** : East Elmhurst (33 %, voisine de LaGuardia), puis des zones à plus faible volume (28-29 %)

**Critère de succès** : l'IC 95 % de la différence de WAPE avec la référence doit exclure 0

## Étape 5 : LightGBM et MLflow

**4 essais** (un seul choix change à la fois) : perte L2, L1, Poisson ; Poisson sans météo
**1er essai** (learning_rate 0,05, 5 000 arbres max) : arrêt anticipé jamais déclenché
**2e essai** (learning_rate 0,1, 20 000 arbres max) : arrêt entre 1 408 et 4 626 arbres, scores inchangés

**Résultats (validation)**
- référence moyenne 4 semaines : WAPE 20,1 %
- LightGBM : 14,1 % (L1, Poisson, Poisson sans météo), 14,6 % (L2)
- écart avec la référence : environ -6 points, IC 95 % [-7,6 ; -4,7], 56 zones sur 56 améliorées
- météo : apport nul, IC [-0,2 ; +0,2] point (et pourtant météo observée, donc optimiste)

**Modèle retenu** : Poisson sans météo, WAPE 14,1 % [13,4 ; 14,9] (le plus simple, sans prévision météo)

**Importance (gain)** : demande passée au même créneau dominante (68 % puis 90 % selon l'essai)
- instable selon les réglages : à confirmer avec SHAP ou permutation
**Bruit incompressible** (Poisson, approximatif) : environ 9 % de WAPE

## Étape 6 : LSTM avec PyTorch

**Modèle** : un exemple par (zone, jour) ; entrée = 168 heures d'historique de la zone,
calendrier du jour, référence des 24 heures, embedding de zone ; sortie = 24 heures
- prédit une correction de la référence ; normalisation par zone (moyenne de l'entraînement)
- perte : erreur absolue sur la demande réelle (alignée sur la WAPE) ; 68 184 paramètres
- 50 680 exemples d'entraînement, 5 152 de validation ; environ 42 s par époque sur CPU

**Entraînement** : 40 époques, meilleure à la 38e, arrêt anticipé jamais déclenché
(courbe presque plate : -0,2 point entre les époques 21 et 38)

**Résultats (validation)**
- LSTM : WAPE 15,1 % [14,4 ; 15,9]
- contre la référence : [-6,6 ; -3,7] points, meilleur
- contre LightGBM : [+0,7 ; +1,2] point, LightGBM réellement meilleur
- moyenne LightGBM + LSTM : 14,1 %, aucun gain [-0,1 ; +0,1]

**Décision** : LightGBM reste le modèle principal (plus précis, environ 2 min contre 28, plus simple)
**Pistes non testées** : 14 jours d'historique, plus d'époques, baisse progressive du learning rate

## Étape 7 : détection d'anomalies

**Méthode** : résidu (réel - prévision LightGBM) / racine(prévision + 1), puis score robuste
par zone (médiane et MAD) ; seuil |score| > 4 ; agrégation à l'échelle de la ville et
regroupement des heures consécutives en épisodes

**Zone x heure (validation)** : 1 267 heures anormales au seuil 4 (1,02 %), dont 1 063 au-dessus
de la prévision ; queues épaisses (une loi normale donnerait environ 0,006 %)

**Ville** : 53 heures anormales, 23 épisodes

**Événements listés avant l'analyse**
- inondations du 14 juillet : détecté, épisode n° 1 (17 h-21 h, +17 334 trajets, 42 zones)
- 4 juillet : détecté, demande plus faible que prévu (score -4,4)
- Labor Day : non détecté (le modèle le prévoit déjà grâce à la variable jour férié)

**Explications après coup (hypothèses à vérifier)**
- aéroports la nuit (LaGuardia 1 h-2 h, JFK 2 h) : vols retardés ?
- 2 et 3 juillet en soirée, demande plus faible : départs avant le week-end du 4 juillet
  -> piste : variable « veille de jour férié »
- 13 août et 20 août en fin de journée : à vérifier avec la pluie
- le fichier d'événements a été rempli après le premier lancement, mais avec les 3 événements
  proposés avant l'analyse ; aucun événement ajouté après coup

  **Lien avec la pluie (après coup)** : 6 des 8 épisodes au-dessus de la prévision ont eu
une heure de pluie > 0,8 mm/h (95e centile d'été) ; les 2 creux (2 et 3 juillet) par temps sec
- taux de base des fenêtres pluvieuses : [à compléter avec check_rain_baseline.py]
- la météo n'aide pas en moyenne (étape 5) mais semble expliquer les heures extrêmes, rares
- limite : prévisions de pluie heure par heure peu fiables la veille ; un seul point de mesure

- taux de base des fenêtres pluvieuses : 7 % (3 h), 9 % (5 h), 12 % (8 h), contre 75 % pour les
  épisodes au-dessus de la prévision ; probabilité d'au moins 6 sur 8 par hasard : environ 1 sur 15 000
- à confirmer sur une autre période (le test, à la fin du projet)