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
| Sortie | `output/feature_dataset.parquet`, **~700 features** agrégées, clé `SK_ID_CURR` |
| Encodage | one-hot des catégorielles + ratios métier (`PAYMENT_RATE`, `INCOME_CREDIT_PERC`, …) |

⚠️ Point structurant pour l'API : **le modèle ne consomme pas les données brutes d'un
client**, mais un vecteur de ~700 features agrégées sur son historique multi-tables.
Le contrat d'entrée de l'API est donc une décision de conception à part entière
(étape 2).

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
