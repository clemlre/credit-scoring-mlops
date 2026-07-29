# Credit Scoring — mise en production (MLOps)

Mise en production du modèle de scoring de crédit de **« Prêt à Dépenser »** : API de
prédiction, conteneurisation, CI/CD, et suivi du modèle en production (data drift,
latence, optimisation).

> Projet OpenClassrooms **P8 — Confirmez vos compétences en MLOps (2/2)**.
> Le modèle servi ici provient du projet précédent (*Initiez-vous au MLOps*, 1/2) :
> voir [`docs/modele-partie1.md`](docs/modele-partie1.md).

## En deux lignes

Un `LGBMClassifier` prédit la probabilité de défaut d'un demandeur de crédit à partir
de 779 features agrégées sur son historique. La décision d'octroi se prend au **seuil
0,10** (et non 0,5), parce que la métrique métier pénalise un mauvais client accepté
**10 fois** plus qu'un bon client refusé.

## État d'avancement

| Étape | Contenu | Statut |
|---|---|---|
| 1 | Contrôle de version, structure du projet, documentation initiale | ✅ en place |
| 2 | API de prédiction, tests, Dockerfile, pipeline CI/CD | ✅ en place |
| 3 | Stockage des données de production + analyse du data drift | ⬜ à venir |
| 4 | Profiling et optimisation des performances d'inférence | ⬜ à venir |

## Structure du dépôt

```
.
├── api/                  # code de l'API de prédiction — SEUL code déployé
│   ├── config.py         #   réglages lus depuis l'environnement
│   ├── model.py          #   chargement du modèle et inférence
│   ├── schemas.py        #   contrat d'entrée/sortie (Pydantic → Swagger)
│   └── main.py           #   routes et gestion des erreurs
├── src/                  # pipeline de features + entraînement (hérité de la Partie 1)
│   └── export_model.py   #   pont MLflow → artefact déployable
├── scripts/smoke_test.py # vérifie un service qui tourne (conteneur, déploiement)
├── notebooks/            # analyses Partie 1, puis notebook de data drift (étape 3)
├── tests/                # tests automatisés pytest
├── models/               # artefact déployable + paramètres de référence
├── monitoring/           # logs de production et rapports de drift (étape 3)
├── docs/                 # documentation et captures d'écran
├── data/                 # CSV Home Credit — NON versionnés, voir data/README.md
├── .github/workflows/    # pipeline CI/CD
├── Dockerfile            # image de l'API (multi-étapes, utilisateur non-root)
└── pyproject.toml        # dépendances, gérées avec uv
```

## Installation

Prérequis : **Python 3.11** et [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/clemlre/credit-scoring-mlops.git
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

### Préparer l'artefact de modèle

L'artefact déployable (`models/credit_default_lgbm.txt`) est **versionné** : l'API,
les tests et l'image Docker fonctionnent sans rien préparer. Pour le régénérer depuis
le registre MLflow de la Partie 1 :

```bash
uv run --group training python src/export_model.py
uv run --group training python src/export_model.py --p6-root "/chemin/vers/projet-partie-1"
```

Le script vérifie lui-même que l'artefact reproduit le modèle du registre à
l'identique sur 500 clients, et refuse d'aboutir sinon.

### En local

```bash
uv sync
uv run uvicorn api.main:app --reload
```

- Documentation interactive (Swagger) : <http://127.0.0.1:8000/docs>
- Sonde de disponibilité : <http://127.0.0.1:8000/health>

### Avec Docker

```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
python scripts/smoke_test.py http://127.0.0.1:8000   # vérifie le service qui tourne
```

### Les routes

| Route | Rôle |
|---|---|
| `GET /health` | Disponibilité. Répond `503` tant que le modèle n'est pas chargé. |
| `GET /model/info` | Version, provenance, performances et règles d'acceptation. |
| `GET /features` | Contrat d'entrée : les 779 features acceptées, par origine. |
| `POST /predict` | Score une demande de crédit. |
| `POST /predict/batch` | Score un lot de demandes en un seul appel au modèle. |

### Que faut-il envoyer ?

Le modèle ne consomme pas les données brutes d'un client, mais **779 features
agrégées** sur son historique. L'API accepte un sous-ensemble libre de ces features :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"AMT_CREDIT": 406597.5, "EXT_SOURCE_2": 0.2629, ...}}'
```

```json
{
  "probability": 0.0731,
  "decision": "accepted",
  "threshold": 0.1,
  "model_version": "1",
  "coverage": {"features_provided": 583, "features_missing": 196,
               "application_ratio": 0.887, "history_ratio": 0.686}
}
```

**Comment lire la réponse.** `probability` est la probabilité de **défaut**. Elle est
comparée au seuil **0,10** — et non 0,5 — parce que le coût métier pénalise un mauvais
client accepté dix fois plus qu'un bon client refusé. `coverage` dit sur quelle
quantité d'information la décision a été prise : une probabilité calculée sur un
dossier à moitié vide ne se lit pas comme une probabilité calculée sur un dossier
complet.

### Ce que l'API refuse, et pourquoi

| Situation | Réponse | Raison |
|---|---|---|
| Nom de feature inconnu | `422` | Une faute de frappe absorbée en silence donnerait un score calculé sans la variable qu'on croyait fournir. |
| Valeur hors plage | `422` | Bornes mesurées sur les 307 507 clients d'entraînement — une valeur en dehors ne peut pas venir d'un dossier réel. |
| Dossier trop incomplet | `422` | Sous 50 % des features de demande renseignées, le score n'a plus de valeur métier. L'historique de crédit, lui, reste facultatif. |
| Type incorrect, `Infinity`, `NaN` | `422` | Une chaîne `"0,5"` ou une valeur non finie fausseraient le calcul sans erreur visible. |
| Modèle non chargé | `503` | Le service le dit explicitement au lieu de redémarrer en boucle. |

### Réglages

| Variable | Défaut | Effet |
|---|---|---|
| `MODEL_DIR` | `models/` | Emplacement de l'artefact de modèle. |
| `MIN_APPLICATION_COVERAGE` | `0.5` | Part minimale du dossier de demande exigée. |
| `MAX_BATCH_SIZE` | `1000` | Plafond du mode lot. |
| `PORT` | `8000` | Port d'écoute (utilisé par les hébergeurs). |

## Tests

```bash
uv run pytest                                    # suite complète
uv run pytest --cov=api --cov-report=term-missing  # avec couverture
```

68 tests, **100 % de couverture** sur `api/` (plancher CI : 95 %). Aucune donnée
client n'est versionnée : les dossiers de test sont générés de façon déterministe.
Les tests de fidélité numérique sur de vrais clients s'ignorent d'eux-mêmes si les
données de la Partie 1 ne sont pas disponibles.

## Intégration et déploiement continus

`.github/workflows/ci.yml`, déclenché sur push `main`, sur pull request vers `main`,
et manuellement.

1. **Lint et tests** — `ruff`, puis `pytest` avec plancher de couverture.
2. **Image Docker** — construction, démarrage du conteneur, et test de fumée contre
   le service réel. L'image n'est publiée sur GHCR que si ce test passe, et jamais
   depuis une pull request.
3. **Déploiement** — uniquement depuis `main`. S'active si le secret `HF_TOKEN` et la
   variable `HF_SPACE` sont définis, puis revérifie le service déployé.

Aucun identifiant n'est écrit dans le dépôt : la publication d'image utilise le
`GITHUB_TOKEN` éphémère, le déploiement un secret de dépôt.

La procédure de configuration du déploiement est décrite dans
[`docs/deploiement.md`](docs/deploiement.md).

## Pourquoi ces choix techniques

FastAPI plutôt que Gradio, format texte natif plutôt que pickle, bornes de validation
mesurées plutôt que décrétées : chaque décision est justifiée — avec ce qui a été écarté
et à quelle condition elle deviendrait mauvaise — dans
[`docs/choix-techniques.md`](docs/choix-techniques.md).

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
