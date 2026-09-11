# Fiche modèle — héritage de la Partie 1

Ce document décrit **le modèle qu'on met en production** dans ce dépôt. Il a été
développé, versionné et évalué au projet précédent (*Initiez-vous au MLOps*, 1/2) et
constitue le point de départ, non l'objet, de ce projet-ci.

## Problème

Scoring de défaut de crédit — dataset [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).
Cible binaire `TARGET` : 1 = le client a fait défaut. Classes très déséquilibrées
(~8 % de positifs).

## Données et features

| | |
|---|---|
| Tables sources | 7 CSV (`application_train/test`, `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `installments_payments`, `credit_card_balance`) |
| Pipeline | `src/prepare_data.py` — adapté du kernel Kaggle *jsaguiar* |
| Sortie | `output/feature_dataset.parquet`, **779 features** agrégées, clé `SK_ID_CURR` |
| Encodage | one-hot des catégorielles + ratios métier (`PAYMENT_RATE`, `INCOME_CREDIT_PERC`, …) |

⚠️ Point structurant pour l'API : **le modèle ne consomme pas les données brutes d'un
client**, mais un vecteur de 779 features agrégées sur son historique multi-tables.

Ce contrat a été tranché à l'étape 2 : l'API accepte un sous-ensemble libre de ces
779 features, réparties en **245 features « dossier de demande »** (renseignées par le
demandeur) et **534 agrégats d'historique de crédit** (calculés sur ses crédits passés,
donc absents pour un primo-emprunteur). Voir le README, section « Que faut-il envoyer ».

## Modèle

`LGBMClassifier` (LightGBM 4.6.0), hyperparamètres cherchés par **Optuna** (30 essais,
sampler TPE, `MedianPruner`) sur un sous-échantillon stratifié de 50 000 lignes, puis
modèle final réentraîné sur l'intégralité du train.

```json
{
  "learning_rate": 0.0308, "num_leaves": 96, "max_depth": 8,
  "min_child_weight": 45.32, "min_child_samples": 94,
  "subsample": 0.749, "colsample_bytree": 0.788,
  "reg_alpha": 1.094, "reg_lambda": 0.0104, "min_split_gain": 0.0675,
  "n_estimators": 867
}
```

Valeurs de référence : [`models/optuna_best_params.json`](../models/optuna_best_params.json).

## Métrique métier et seuil de décision

Le coût métier pilote **tout** le projet :

```
business_cost = 10 × FN + 1 × FP
```

Un mauvais client accepté (FN) fait perdre le capital prêté ; un bon client refusé (FP)
ne fait perdre que les intérêts. D'où le facteur 10.

Conséquence directe : **le seuil de décision n'est pas 0,5 mais 0,10**, obtenu par
balayage out-of-fold ([`models/threshold_sweep.csv`](../models/threshold_sweep.csv)).

| Seuil | Coût métier OOF |
|---|---|
| 0,05 | 162 228 |
| **0,10** | **150 877** ← optimum |
| 0,50 | 236 407 |

Soit **−36 %** de coût par rapport au seuil naïf de 0,5. L'API doit exposer ce seuil
explicitement : renvoyer une probabilité sans dire à quel seuil elle se compare n'a
aucune valeur métier.

Performance discriminante : **AUC OOF = 0,789**.

### Métriques out-of-fold du modèle servi

Rejouées dans ce dépôt avec le protocole de la Partie 1 (`scripts/evaluer_modele.py` :
3 plis stratifiés, graine 42, 307 507 clients étiquetés). Résultats dans
`docs/perf/metriques-modele.json` et dans `models/model_metadata.json`, donc exposés par
`GET /model/info`.

| | Seuil 0,10 (métier) | Seuil 0,50 (naïf) |
|---|---:|---:|
| AUC (indépendante du seuil) | 0,7888 | 0,7888 |
| Accuracy | 78,3 % | 92,0 % |
| Précision (vrais défauts / dossiers refusés) | 21,2 % | 56,9 % |
| Rappel (défauts détectés / défauts réels) | 62,3 % | 5,2 % |
| F1 | 0,317 | 0,096 |
| Coût métier | 150 981 | 236 227 |
| Taux de refus | 23,7 % | 0,7 % |
| VP / FP / FN / VN | 15 476 / 57 491 / 9 349 / 225 191 | 1 301 / 987 / 23 524 / 281 695 |

Lecture : avec 8 % de défauts, accepter tout le monde donne déjà 92 % d'accuracy. Le seuil
0,50 « gagne » sur l'accuracy en n'attrapant que 5 % des défauts ; le seuil 0,10 en attrape
62 % et refuse 24 % des dossiers. Les métriques qui comptent ici sont le rappel des défauts
et le coût métier ; l'accuracy n'en est pas une. Le même script montre que la conversion
ONNX et les optimisations de l'étape 4 ne déplacent aucune de ces métriques (voir
[`optimisation.md`](optimisation.md)).

## Traçabilité MLflow (Partie 1)

| | |
|---|---|
| Backend | SQLite `mlruns.db` + artefacts locaux `mlartifacts/` |
| Expérience | `credit-default` |
| Modèle enregistré | `credit-default-lgbm` (Model Registry), loggé via `mlflow.lightgbm.log_model` |

⚠️ Ni `mlruns.db`, ni `mlartifacts/`, ni les données ne sont versionnés ici (voir
`.gitignore`) : ils vivent dans le dépôt de la Partie 1. Produire un **artefact
sérialisé déployable** à partir de ce registre est le premier chantier de l'étape 2.

## Code hérité

| Fichier | Rôle |
|---|---|
| `src/prepare_data.py` | agrégation des 7 tables → parquet de features |
| `src/training.py` | setup MLflow, chargement des données, `business_cost`, boucle CV |
| `src/optimize_lgbm.py` | recherche Optuna, balayage de seuil OOF, enregistrement au registry |
| `src/run_step2_baselines.py`, `run_step3_models.py`, `run_mlp_activations.py` | comparaisons de modèles de la Partie 1 |
| `notebooks/01` → `06` | analyses : préparation, MLflow, expérimentations, optimisation, activations MLP, importance des features (SHAP) |
