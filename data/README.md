# `data/` — données brutes Home Credit (non versionnées)

Ce dossier est **volontairement vide dans Git** (`data/*` est ignoré) : les CSV bruts
pèsent ~2,6 Go et sont soumis aux conditions d'utilisation Kaggle. Un dépôt public
n'est pas un entrepôt de données.

## Récupérer les données

Source : [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) (Kaggle).

Placer les 7 tables utilisées par le pipeline de features à la racine de `data/` :

| Fichier | Rôle |
|---|---|
| `application_train.csv` | demandes de crédit + `TARGET` (défaut = 1) |
| `application_test.csv` | demandes sans label (jeu de soumission Kaggle) |
| `bureau.csv` | crédits déclarés au bureau de crédit |
| `bureau_balance.csv` | historique mensuel de ces crédits |
| `previous_application.csv` | demandes précédentes chez Home Credit |
| `POS_CASH_balance.csv` | historique des crédits POS / cash |
| `installments_payments.csv` | échéances et paiements réels |
| `credit_card_balance.csv` | historique des cartes de crédit |

Puis :

```bash
uv run python src/prepare_data.py          # ~700 features -> output/feature_dataset.parquet
uv run python src/prepare_data.py --debug  # 10 000 lignes par table, pour tester vite
```

Le parquet produit (`output/feature_dataset.parquet`, ~280 Mo) est lui aussi ignoré :
il est **reproductible** à partir des CSV et du code versionné.
