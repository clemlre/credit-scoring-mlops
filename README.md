# Credit Scoring — mise en production (MLOps)

Mise en production du modèle de scoring de crédit de **« Prêt à Dépenser »** : API de
prédiction, conteneurisation, CI/CD, et suivi du modèle en production (data drift,
latence, optimisation).

> Projet OpenClassrooms **P8 — Confirmez vos compétences en MLOps (2/2)**.
> Le modèle servi ici provient du projet précédent (*Initiez-vous au MLOps*, 1/2) :
> voir [`docs/modele-partie1.md`](docs/modele-partie1.md).

## En deux lignes

Un `LGBMClassifier` prédit la probabilité de défaut d'un demandeur de crédit à partir
de ~700 features agrégées sur son historique. La décision d'octroi se prend au **seuil
0,10** (et non 0,5), parce que la métrique métier pénalise un mauvais client accepté
**10 fois** plus qu'un bon client refusé.

## État d'avancement

| Étape | Contenu | Statut |
|---|---|---|
| 1 | Contrôle de version, structure du projet, documentation initiale | ✅ en place |
| 2 | API de prédiction, tests, Dockerfile, pipeline CI/CD | ⬜ à venir |
| 3 | Stockage des données de production + analyse du data drift | ⬜ à venir |
| 4 | Profiling et optimisation des performances d'inférence | ⬜ à venir |

## Structure du dépôt

```
.
├── api/                  # code de l'API de prédiction (étape 2)
├── src/                  # pipeline de features + entraînement (hérité de la Partie 1)
├── notebooks/            # analyses Partie 1, puis notebook de data drift (étape 3)
├── tests/                # tests automatisés pytest (étape 2)
├── models/               # paramètres de référence du modèle ; artefacts sérialisés (non versionnés)
├── monitoring/           # logs de production et rapports de drift (étape 3)
├── docs/                 # documentation et captures d'écran
├── data/                 # CSV Home Credit — NON versionnés, voir data/README.md
├── .github/workflows/    # pipeline CI/CD (étape 2)
└── pyproject.toml        # dépendances, gérées avec uv
```

## Installation

Prérequis : **Python 3.11** et [uv](https://docs.astral.sh/uv/).

```bash
git clone <url-du-depot>
cd credit-scoring-mlops

uv sync                          # coeur d'inférence + outils de dev (uv inclut le groupe `dev` par défaut)
uv sync --no-dev                 # coeur d'inférence seul — ce que contiendra l'image Docker
uv sync --group training         # + MLflow, Optuna, SHAP, Jupyter (reproduire la Partie 1)
```

Les dépendances sont volontairement séparées : `[project].dependencies` ne contient que
ce qui est nécessaire pour **charger le modèle et prédire**, c'est-à-dire ce qui partira
dans l'image Docker. Tout le reste (entraînement, notebooks, tests) vit dans des groupes
optionnels.

### Reproduire le dataset de features

Les données brutes ne sont pas versionnées (~2,6 Go, licence Kaggle) — voir
[`data/README.md`](data/README.md). Une fois les CSV en place :

```bash
uv run --group training python src/prepare_data.py
```

## Lancer l'API

> _Section à compléter à l'étape 2._

## Interpréter le monitoring

> _Section à compléter à l'étape 3._

## Conventions de travail

**Branches**

| Branche | Rôle |
|---|---|
| `main` | état déployable ; c'est elle qui déclenche la CI/CD |
| `feat/<sujet>` | développement d'une fonctionnalité, fusionnée dans `main` par pull request |
| `fix/<sujet>` | correction de bug |

**Messages de commit** — convention [Conventional Commits](https://www.conventionalcommits.org/fr/) :
`type(portée): description à l'infinitif`, avec `feat`, `fix`, `docs`, `test`, `ci`,
`chore`, `refactor`, `perf`.

**Ce qui n'entre jamais dans le dépôt** : données clients, CSV bruts, artefacts MLflow,
secrets et credentials (voir `.gitignore`).

## Licence

[MIT](LICENSE) — le code uniquement. Les données Home Credit restent soumises aux
conditions d'utilisation de Kaggle.
