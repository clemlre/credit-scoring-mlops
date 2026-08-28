# Plan d'implémentation — Notebook d'analyse du data drift

> **Pour un exécutant agentique :** SOUS-COMPÉTENCE REQUISE — utiliser
> `superpowers:subagent-driven-development` (recommandé) ou
> `superpowers:executing-plans` pour dérouler ce plan tâche par tâche. Les étapes
> utilisent la syntaxe à cases (`- [ ]`) pour le suivi.

**But :** produire `notebooks/07_data_drift.ipynb`, livrable 6 du projet, qui mesure la
dérive des données entre le jeu d'entraînement de la Partie 1 et les prédictions
réellement rendues en production, et qui rapporte les métriques opérationnelles du
service.

**Architecture :** tout le code d'analyse vit dans le notebook — pas de module Python.
Deux sources : le parquet de features de la Partie 1 (référence, hors dépôt) et la table
`predictions` de PostgreSQL (courant, deux fenêtres temporelles). Evidently calcule les
PSI ; pandas et matplotlib font le reste.

**Pile technique :** Python 3.11, Evidently 0.7.21, pandas 2.3.3, psycopg 3.2.12,
LightGBM 4.6.0, matplotlib, uv.

**Spécification :** `docs/specs/2026-08-28-analyse-data-drift.md` — à lire avant de
commencer. Ce plan applique cette spec ; les deux voyagent ensemble.

## Contraintes globales

Copiées mot pour mot de la spec. Elles s'appliquent implicitement à **chaque** tâche.

- **Tout le code d'analyse vit dans le `.ipynb`.** Ne créer aucun package `monitoring/`,
  aucun module importé par le notebook.
- **Rédaction sobre.** Cellules markdown courtes disant ce qu'on regarde et ce qu'on en
  conclut. Commentaires rares et utiles uniquement. Aucun ton didactique, aucune
  formule d'accompagnement, aucun emoji. Le notebook doit se lire comme le travail d'un
  data scientist, pas comme un tutoriel.
- **Aucune donnée client versionnée.** Pas d'extrait CSV/parquet, pas de rapport
  Evidently HTML committé. Les résultats vivent dans les sorties sauvegardées du
  notebook.
- **Aucune donnée effacée.** Ni `docker compose down -v`, ni `TRUNCATE`. Le nouveau
  trafic s'ajoute à l'existant ; le découpage se fait par fenêtres temporelles.
- **Evidently reste hors du cœur d'inférence.** Groupe de dépendances `monitoring`
  uniquement. `api/` ne doit jamais l'importer, l'image Docker reste à 583 Mo.
- **Méthode de dérive : PSI.** Bandes retenues : `< 0,10` stable · `0,10–0,25` modérée ·
  `> 0,25` significative.
- **Référence :** échantillon de 20 000 lignes, `random_state=42`, pris parmi les lignes
  du parquet où `TARGET` est renseigné.
- **Top features suivies :** 20, triées par `importance_type="gain"` lue dans le booster
  LightGBM servi.
- **Langue :** français.

## Adaptation du cycle de test

Ce plan ne peut pas suivre un cycle TDD classique : la décision projet place tout le
code dans un notebook, donc il n'existe aucun module que `pytest` puisse importer.
L'adaptation retenue, appliquée à chaque tâche de contenu :

1. **Prototyper** le calcul dans un script jetable (`scratch/<nom>.py`), l'exécuter
   contre les vraies sources, et **comparer la sortie à une attente écrite à l'avance**.
2. Seulement une fois la sortie conforme, **reporter le code dans une cellule** du
   notebook.
3. Exécuter le notebook et vérifier que la cellule produit la même sortie.

Le répertoire `scratch/` est temporaire et n'est jamais committé. C'est cette étape 1
qui joue le rôle du test : sans elle, on écrirait du code dans un notebook sans jamais
en vérifier le résultat.

La tâche 4 installe en plus une **vérification permanente** dans le notebook lui-même :
deux cas de contrôle sur l'instrument de mesure, qui restent exécutés à chaque lancement.

## Structure des fichiers

| Fichier | Responsabilité | Action |
|---|---|---|
| `notebooks/07_data_drift.ipynb` | L'analyse complète : chargement, contrôles, dérive, métriques, conclusions | Créer |
| `pyproject.toml` | Déclare le groupe de dépendances `monitoring` | Modifier |
| `uv.lock` | Résolution figée | Régénérer |
| `.gitignore` | Exclut les exports HTML de rapports | Modifier |
| `README.md` | Renvoi au notebook ; correction de la description de `monitoring/` | Modifier |
| `scratch/` | Prototypes jetables, jamais committés | Créer puis supprimer |

---

## Tâche 1 : Dépendance Evidently et vérification de l'API

**Fichiers :**
- Modifier : `pyproject.toml` (section `[dependency-groups]`)
- Régénérer : `uv.lock`
- Créer puis supprimer : `scratch/verif_evidently.py`

**Interfaces :**
- Consomme : rien
- Produit : le motif d'appel Evidently vérifié, réutilisé par les tâches 4, 6 et 7 —
  en particulier la fonction `psi_par_colonne(reference, courant, colonnes) -> pd.Series`
  définie à l'étape 5 ci-dessous.

- [ ] **Étape 1 : Écrire le script de vérification avec ses attentes**

Créer `scratch/verif_evidently.py` :

```python
"""Vérifie le motif d'appel Evidently sur des données de contrôle.

Attentes, établies sur un tirage déterministe :
  - colonne décalée de 0,8 sigma -> PSI > 0,25 (bande "significative")
  - colonne non décalée          -> PSI < 0,10 (bande "stable")
"""

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def psi_par_colonne(reference, courant, colonnes):
    """PSI par colonne, trié du plus dérivé au moins dérivé."""
    colonnes = list(colonnes)
    rapport = Report([DataDriftPreset(method="psi", columns=colonnes)])
    resultat = rapport.run(
        current_data=courant[colonnes], reference_data=reference[colonnes]
    ).dict()

    valeurs = {}
    for metrique in resultat["metrics"]:
        config = metrique.get("config", {})
        if config.get("type", "").endswith("ValueDrift"):
            valeurs[config["column"]] = metrique["value"]
    return pd.Series(valeurs).sort_values(ascending=False)


rng = np.random.default_rng(0)
reference = pd.DataFrame({"decalee": rng.normal(0, 1, 2000), "stable": rng.normal(5, 2, 2000)})
courant = pd.DataFrame({"decalee": rng.normal(0.8, 1, 800), "stable": rng.normal(5, 2, 800)})

psi = psi_par_colonne(reference, courant, ["decalee", "stable"])
print(psi)

assert psi["decalee"] > 0.25, "la colonne décalée devrait sortir en dérive significative"
assert psi["stable"] < 0.10, "la colonne stable devrait rester sous le seuil"
print("\nMotif d'appel Evidently vérifié.")
```

- [ ] **Étape 2 : Ajouter le groupe de dépendances**

Dans `pyproject.toml`, à la fin de la section `[dependency-groups]`, après le groupe
`dev` :

```toml
# Analyse du data drift (étape 3). Hors du cœur d'inférence : l'API n'importe jamais
# Evidently, et l'image Docker n'en contient pas une ligne.
monitoring = [
    "evidently==0.7.21",
]
```

- [ ] **Étape 3 : Résoudre et installer**

```bash
uv lock
uv sync --group training --group monitoring
```

`training` est nécessaire en plus : il apporte `jupyter`, `pandas`, `pyarrow` et
`matplotlib`, dont le notebook a besoin pour lire le parquet, tracer et s'exécuter.

Attendu : `uv lock` ajoute `evidently`, `plotly` et leurs dépendances sans conflit.

- [ ] **Étape 4 : Exécuter la vérification**

```bash
uv run python scratch/verif_evidently.py
```

Attendu — les deux assertions passent et la sortie ressemble à :

```
decalee    0.791482
stable     0.031949
dtype: float64

Motif d'appel Evidently vérifié.
```

Si une assertion échoue, ne pas continuer : la suite du plan repose entièrement sur ce
motif d'appel.

- [ ] **Étape 5 : Vérifier que l'API n'est pas contaminée**

```bash
grep -rn "evidently" api/ && echo "PROBLEME : api/ importe evidently" || echo "OK : api/ est propre"
uv run pytest -q
```

Attendu : `OK : api/ est propre`, puis `95 passed, 5 skipped` (ou `100 passed` si la
pile PostgreSQL tourne et que `DATABASE_URL` est défini).

- [ ] **Étape 6 : Committer**

```bash
git add pyproject.toml uv.lock
git commit -m "build: ajouter le groupe de dépendances monitoring (Evidently)

Evidently 0.7.21, hors du cœur d'inférence : l'API ne l'importe pas et
l'image Docker reste inchangée. Résolution vérifiée sans conflit avec
numpy<2.3, pandas, scikit-learn, MLflow et SHAP."
```

---

## Tâche 2 : Génération du trafic et relevé des bornes de fenêtres

**Fichiers :**
- Aucun fichier du dépôt modifié. Cette tâche produit des **données** et **trois
  horodatages** que la tâche 3 inscrira dans le notebook.

**Interfaces :**
- Consomme : `scripts/simuler_trafic.py` (déjà présent)
- Produit : les constantes `DEBUT_ANALYSE`, `COUPURE`, `FIN_ANALYSE` (chaînes ISO 8601
  avec fuseau), consommées par la tâche 3.

- [ ] **Étape 1 : Démarrer la pile et relever la borne de départ**

```bash
API_PORT=8001 docker compose up -d
docker exec scoring-db psql -U scoring -d monitoring -tAc "SELECT max(occurred_at) FROM predictions;"
```

Noter cette valeur : c'est **`DEBUT_ANALYSE`**. Toutes les lignes antérieures
proviennent de runs de mise au point et seront exclues.

Vérifier au passage que l'API répond et que la base est branchée :

```bash
curl -s http://127.0.0.1:8001/health
```

Attendu : `"prediction_log":{"stdout":true,"database":"ready","last_error":null}`

- [ ] **Étape 2 : Générer la fenêtre A (trafic normal)**

```bash
uv run python scripts/simuler_trafic.py --url http://127.0.0.1:8001 --nombre 3000 --lot 200
```

Attendu : `3000 dossiers réels tirés du jeu de la Partie 1.` puis
`3000 prédictions journalisées — <n> acceptées, <m> refusées.`

Si le message indique un repli synthétique, **s'arrêter** : le parquet de la Partie 1
est introuvable et l'analyse de dérive n'aurait aucune valeur. Définir
`P6_PROJECT_ROOT` vers le projet P6 et relancer.

Si un lot est refusé pour cause de taille de requête, relancer avec `--lot 100`.

- [ ] **Étape 3 : Relever la coupure et marquer une pause**

```bash
docker exec scoring-db psql -U scoring -d monitoring -tAc "SELECT max(occurred_at) FROM predictions;"
```

Noter cette valeur : c'est **`COUPURE`**. Attendre ensuite deux minutes avant la suite,
pour que la frontière entre les deux fenêtres soit franche et lisible dans le tableau
des volumes par minute.

- [ ] **Étape 4 : Générer la fenêtre B (trafic décalé)**

```bash
uv run python scripts/simuler_trafic.py --url http://127.0.0.1:8001 --nombre 1000 --lot 200 --decalage 0.2
```

Attendu : le taux de refus est nettement supérieur à celui de la fenêtre A. Sur les
mesures effectives : 20,7 % de refus en trafic normal contre 45,0 % à `--decalage 0.2`.

- [ ] **Étape 5 : Relever la borne de fin et contrôler les volumes**

```bash
docker exec scoring-db psql -U scoring -d monitoring -tAc "SELECT max(occurred_at) FROM predictions;"

docker exec scoring-db psql -U scoring -d monitoring -c "
SELECT date_trunc('minute', occurred_at) AS minute, count(*)
FROM predictions GROUP BY 1 ORDER BY 1;"
```

Noter la première valeur : c'est **`FIN_ANALYSE`**.

Attendu dans le second tableau : un trou d'au moins une minute entre la fin de la
fenêtre A et le début de la fenêtre B. Si le trou est absent, les deux fenêtres se
touchent — régénérer la fenêtre B après une vraie pause.

- [ ] **Étape 6 : Consigner les trois bornes**

Écrire les trois horodatages relevés dans un fichier temporaire
`scratch/bornes.txt`, au format exact renvoyé par PostgreSQL, par exemple :

```
DEBUT_ANALYSE = 2026-08-28 09:02:51.337288+00
COUPURE       = 2026-08-28 14:11:07.882145+00
FIN_ANALYSE   = 2026-08-28 14:15:42.019873+00
```

Aucun commit : cette tâche ne modifie aucun fichier du dépôt.

---

## Tâche 3 : Notebook § 1 — chargement des deux sources

**Fichiers :**
- Créer : `notebooks/07_data_drift.ipynb`
- Créer puis supprimer : `scratch/charger.py`

**Interfaces :**
- Consomme : les trois bornes de la tâche 2
- Produit : trois DataFrames disponibles pour toutes les sections suivantes —
  `reference` (20 000 × 779, colonnes = noms de features du modèle),
  `fenetre_a` et `fenetre_b` (colonnes de features + `occurred_at`, `probability`,
  `decision`, `latency_ms`, `application_ratio`, `history_ratio`).

- [ ] **Étape 1 : Prototyper le chargement avec ses attentes**

Créer `scratch/charger.py` :

```python
"""Prototype du chargement des deux sources.

Attentes :
  - référence : exactement 20 000 lignes, 779 colonnes
  - fenêtre A : ~3 000 lignes ; fenêtre B : ~1 000 lignes
  - les colonnes de features des deux fenêtres sont incluses dans celles du modèle
"""

import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg

RACINE = Path(__file__).resolve().parents[1]
P6 = Path(os.environ.get(
    "P6_PROJECT_ROOT",
    r"C:\Users\ClementLoire\ObsidianVault_MCP_enabled\05 - Cours\P6 - Initiez-vous au MLOps 1-2",
))
PARQUET = P6 / "output" / "feature_dataset.parquet"
DSN = os.environ.get("DATABASE_URL", "postgresql://scoring:scoring_dev@127.0.0.1:5432/monitoring")

DEBUT_ANALYSE = "REMPLACER PAR LA BORNE RELEVEE EN TACHE 2"
COUPURE = "REMPLACER PAR LA BORNE RELEVEE EN TACHE 2"
FIN_ANALYSE = "REMPLACER PAR LA BORNE RELEVEE EN TACHE 2"

TAILLE_REFERENCE = 20_000
GRAINE = 42

booster = lgb.Booster(
    model_str=(RACINE / "models" / "credit_default_lgbm.txt").read_text(encoding="utf-8")
)
NOMS_FEATURES = list(booster.feature_name())

if not PARQUET.exists():
    raise SystemExit(
        f"Parquet de la Partie 1 introuvable : {PARQUET}\n"
        "Définir P6_PROJECT_ROOT vers le projet « Initiez-vous au MLOps 1/2 »."
    )

brut = pd.read_parquet(PARQUET)
reference = (
    brut[brut["TARGET"].notna()]
    .sample(n=TAILLE_REFERENCE, random_state=GRAINE)
    .replace([np.inf, -np.inf], np.nan)
    .reindex(columns=NOMS_FEATURES)
)

COLONNES_SERVICE = [
    "occurred_at", "probability", "decision",
    "latency_ms", "application_ratio", "history_ratio",
]

def charger_fenetre(debut, fin):
    requete = f"""
        SELECT {', '.join(COLONNES_SERVICE)}, features
        FROM predictions
        WHERE occurred_at > %s AND occurred_at <= %s
        ORDER BY occurred_at
    """
    try:
        with psycopg.connect(DSN, connect_timeout=5) as connexion:
            lignes = connexion.execute(requete, (debut, fin)).fetchall()
    except psycopg.OperationalError as erreur:
        raise SystemExit(
            f"Base de monitoring injoignable : {erreur}\n"
            "Démarrer la pile avec : API_PORT=8001 docker compose up -d"
        ) from None

    if not lignes:
        raise SystemExit(
            f"Aucune prédiction entre {debut} et {fin}. "
            "Lancer scripts/simuler_trafic.py (voir tâche 2 du plan)."
        )

    service = pd.DataFrame([ligne[:-1] for ligne in lignes], columns=COLONNES_SERVICE)
    features = pd.DataFrame([ligne[-1] for ligne in lignes]).reindex(columns=NOMS_FEATURES)
    return pd.concat([service, features], axis=1)

fenetre_a = charger_fenetre(DEBUT_ANALYSE, COUPURE)
fenetre_b = charger_fenetre(COUPURE, FIN_ANALYSE)

print(f"référence : {reference.shape[0]} lignes, {reference.shape[1]} colonnes")
print(f"fenêtre A : {len(fenetre_a)} prédictions")
print(f"fenêtre B : {len(fenetre_b)} prédictions")

assert reference.shape == (TAILLE_REFERENCE, len(NOMS_FEATURES))
assert len(fenetre_a) > 2500, "fenêtre A trop petite"
assert len(fenetre_b) > 800, "fenêtre B trop petite"
print("\nChargement conforme.")
```

- [ ] **Étape 2 : Renseigner les bornes et exécuter**

Remplacer les trois `REMPLACER PAR...` par les valeurs de `scratch/bornes.txt`, puis :

```bash
uv run python scratch/charger.py
```

Attendu :

```
référence : 20000 lignes, 779 colonnes
fenêtre A : 3000 prédictions
fenêtre B : 1000 prédictions

Chargement conforme.
```

- [ ] **Étape 3 : Créer le notebook avec ses trois premières cellules**

Créer `notebooks/07_data_drift.ipynb`. Cellule markdown 1 :

```markdown
# Analyse de la dérive des données

Le modèle de scoring a été entraîné sur les dossiers Home Credit de la Partie 1. Depuis
sa mise en service, l'API journalise chaque prédiction rendue en base PostgreSQL, avec
le dossier reçu. Ce notebook compare les deux populations.

Deux fenêtres de production sont examinées :

- **fenêtre A** — trafic nominal ;
- **fenêtre B** — trafic volontairement décalé, pour vérifier que le dispositif de
  détection se déclenche et à partir de quelle amplitude.

Référence : 20 000 dossiers tirés au hasard du jeu d'entraînement.
```

Cellule de code 2 : les imports, constantes et le chargement, repris **à l'identique**
du prototype validé, sans les `assert` ni les `print` de contrôle.

Cellule de code 3 : le tableau des volumes par minute, qui rend les frontières
vérifiables :

```python
volumes = (
    pd.concat([
        fenetre_a.assign(fenetre="A"),
        fenetre_b.assign(fenetre="B"),
    ])
    .groupby([pd.Grouper(key="occurred_at", freq="min"), "fenetre"])
    .size()
    .rename("predictions")
    .reset_index()
)
volumes
```

Cellule markdown 4, à rédiger **après** avoir vu la sortie : une à deux phrases
constatant les volumes et le trou entre les deux fenêtres.

- [ ] **Étape 4 : Exécuter le notebook et vérifier**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_data_drift.ipynb
```

Attendu : exécution sans erreur ; le tableau des volumes montre une interruption d'au
moins une minute entre les fenêtres A et B.

- [ ] **Étape 5 : Committer**

```bash
git add notebooks/07_data_drift.ipynb
git commit -m "feat(monitoring): notebook de drift, chargement des deux sources

Référence : 20 000 dossiers du jeu d'entraînement de la Partie 1, échantillon
déterministe. Courant : les prédictions stockées, découpées en deux fenêtres
par horodatage. Le tableau des volumes par minute rend les frontières
vérifiables plutôt que déclaratives."
```

---

## Tâche 4 : Notebook § 2 — vérification de l'instrument de mesure

**Fichiers :**
- Modifier : `notebooks/07_data_drift.ipynb`
- Créer puis supprimer : `scratch/verif_instrument.py`

**Interfaces :**
- Consomme : `reference` (tâche 3)
- Produit : la fonction `psi_par_colonne(reference, courant, colonnes) -> pd.Series` et
  la fonction `bande(psi) -> str`, utilisées par les tâches 6 et 7.

Le code d'analyse vivant dans le notebook, il échappe à la CI. Cette section est la
contrepartie : elle vérifie que la mesure elle-même est juste, **avant** toute
conclusion. Elle reste dans le notebook et s'exécute à chaque lancement.

- [ ] **Étape 1 : Prototyper les deux cas de contrôle**

Créer `scratch/verif_instrument.py` reprenant le chargement de `scratch/charger.py`,
puis :

```python
from evidently import Report
from evidently.presets import DataDriftPreset

SEUIL_STABLE = 0.10
SEUIL_SIGNIFICATIF = 0.25


def psi_par_colonne(reference, courant, colonnes):
    """PSI par colonne, trié du plus dérivé au moins dérivé."""
    colonnes = list(colonnes)
    rapport = Report([DataDriftPreset(method="psi", columns=colonnes)])
    resultat = rapport.run(
        current_data=courant[colonnes], reference_data=reference[colonnes]
    ).dict()

    valeurs = {}
    for metrique in resultat["metrics"]:
        config = metrique.get("config", {})
        if config.get("type", "").endswith("ValueDrift"):
            valeurs[config["column"]] = metrique["value"]
    return pd.Series(valeurs).sort_values(ascending=False)


def bande(psi):
    if psi < SEUIL_STABLE:
        return "stable"
    if psi < SEUIL_SIGNIFICATIF:
        return "modérée"
    return "significative"


colonnes_test = ["EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT", "DAYS_BIRTH"]

# Contrôle 1 — la référence contre elle-même : la mesure doit être muette.
moitie_1 = reference.sample(frac=0.5, random_state=1)
moitie_2 = reference.drop(moitie_1.index)
temoin_nul = psi_par_colonne(moitie_1, moitie_2, colonnes_test)

# Contrôle 2 — décalage connu : la mesure doit réagir.
decalee = reference.copy()
decalee["EXT_SOURCE_2"] = (decalee["EXT_SOURCE_2"] - 0.20).clip(0, 1)
temoin_decale = psi_par_colonne(reference, decalee, ["EXT_SOURCE_2"])

print("contrôle 1 — référence contre elle-même :")
print(temoin_nul)
print("\ncontrôle 2 — EXT_SOURCE_2 décalée de 0,20 :")
print(temoin_decale)

assert temoin_nul.max() < SEUIL_STABLE, "faux positif : la référence dérive d'elle-même"
assert temoin_decale["EXT_SOURCE_2"] > SEUIL_SIGNIFICATIF, "faux négatif : décalage non vu"
print("\nInstrument vérifié.")
```

- [ ] **Étape 2 : Exécuter**

```bash
uv run python scratch/verif_instrument.py
```

Attendu : les deux `assert` passent. Le contrôle 1 donne des PSI proches de 0 (ordre de
grandeur 0,001 à 0,01) ; le contrôle 2 dépasse nettement 0,25.

Si le contrôle 1 échoue, l'échantillonnage ou l'alignement des colonnes est en cause —
ne pas continuer, tout le reste du notebook serait faux.

- [ ] **Étape 3 : Reporter dans le notebook**

Ajouter une cellule markdown :

```markdown
## Vérification de l'instrument

Avant de mesurer quoi que ce soit, on vérifie que la mesure est fiable. Deux contrôles :
la référence comparée à elle-même doit rester sous le seuil de stabilité, et un décalage
connu de 0,20 sur un score externe doit ressortir en dérive significative.
```

Puis deux cellules de code : la première définissant `psi_par_colonne` et `bande`, la
seconde exécutant les deux contrôles — **en conservant les `assert`**, qui font
office de garde-fou permanent.

- [ ] **Étape 4 : Exécuter le notebook**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_data_drift.ipynb
```

Attendu : aucune erreur, les deux contrôles affichés.

- [ ] **Étape 5 : Committer**

```bash
git add notebooks/07_data_drift.ipynb
git commit -m "feat(monitoring): vérifier l'instrument avant de mesurer la dérive

Le code d'analyse vivant dans le notebook, il échappe à la CI. Deux cas de
contrôle le compensent : la référence contre elle-même doit rester muette,
un décalage connu doit être détecté. Les assertions restent dans le notebook
et s'exécutent à chaque lancement."
```

---

## Tâche 5 : Notebook § 3 — sélection des features suivies

**Fichiers :**
- Modifier : `notebooks/07_data_drift.ipynb`
- Créer puis supprimer : `scratch/features_suivies.py`

**Interfaces :**
- Consomme : `booster` (tâche 3)
- Produit : `FEATURES_SUIVIES` — liste de 20 noms de colonnes, et `importances` —
  `pd.Series` indexée par nom de feature. Consommées par les tâches 6 et 7.

- [ ] **Étape 1 : Prototyper**

Créer `scratch/features_suivies.py` reprenant le chargement, puis :

```python
NB_FEATURES_SUIVIES = 20

importances = (
    pd.Series(booster.feature_importance(importance_type="gain"), index=NOMS_FEATURES)
    .sort_values(ascending=False)
)
part_cumulee = importances.cumsum() / importances.sum()

FEATURES_SUIVIES = importances.head(NB_FEATURES_SUIVIES).index.tolist()

print(f"{len(NOMS_FEATURES)} features au total")
print(f"les {NB_FEATURES_SUIVIES} premières portent "
      f"{part_cumulee.iloc[NB_FEATURES_SUIVIES - 1]:.1%} du gain total\n")
print(importances.head(NB_FEATURES_SUIVIES))

assert len(FEATURES_SUIVIES) == NB_FEATURES_SUIVIES
assert all(f in reference.columns for f in FEATURES_SUIVIES)
assert all(f in fenetre_a.columns for f in FEATURES_SUIVIES)
print("\nSélection conforme.")
```

- [ ] **Étape 2 : Exécuter**

```bash
uv run python scratch/features_suivies.py
```

Attendu : les 20 features sont présentes dans les deux sources ; la part cumulée du gain
est affichée. Noter ce pourcentage, il sert d'argument dans la cellule markdown.

- [ ] **Étape 3 : Reporter dans le notebook**

Cellule markdown :

```markdown
## Features suivies

Le modèle compte 779 features. Les comparer toutes produirait un rapport illisible et,
sur autant de tests, une centaine de dérives apparaîtraient par simple effet du nombre.

L'analyse détaillée porte donc sur les 20 features les plus contributives, mesurées par
le gain dans le booster servi. Une dérive sur une feature de poids négligeable ne
déplace pas le score : trier par importance revient à trier par impact métier. La vue
d'ensemble de la section suivante couvre, elle, l'ensemble des features.
```

Puis la cellule de code sans les `assert` ni les `print`, complétée par un graphique en
barres horizontales des 20 importances.

- [ ] **Étape 4 : Exécuter et committer**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_data_drift.ipynb
git add notebooks/07_data_drift.ipynb
git commit -m "feat(monitoring): restreindre l'analyse aux 20 features les plus contributives

Importance lue dans le booster servi plutôt que recopiée du notebook de la
Partie 1 : la source de vérité est l'artefact déployé. Sur 779 tests, une
centaine se déclencherait par pur effet du nombre."
```

---

## Tâche 6 : Notebook § 4 — baseline, production réelle contre entraînement

**Fichiers :**
- Modifier : `notebooks/07_data_drift.ipynb`
- Créer puis supprimer : `scratch/baseline.py`

**Interfaces :**
- Consomme : `reference`, `fenetre_a`, `FEATURES_SUIVIES`, `psi_par_colonne`, `bande`
- Produit : `psi_baseline` — `pd.Series` indexée par feature, consommée par la tâche 7
  pour comparaison.

- [ ] **Étape 1 : Prototyper**

Créer `scratch/baseline.py` reprenant le chargement, `psi_par_colonne`, `bande`,
`importances` et `FEATURES_SUIVIES`, puis :

```python
# Features exclues de la comparaison : une feature absente d'un des deux jeux est une
# information sur le contrat d'entrée, pas un détail à passer sous silence.
communes = [c for c in NOMS_FEATURES if c in fenetre_a.columns]
exclues = [c for c in NOMS_FEATURES if c not in communes]
vides_en_production = [c for c in communes if fenetre_a[c].isna().all()]

print(f"features du modèle        : {len(NOMS_FEATURES)}")
print(f"features comparables      : {len(communes)}")
print(f"absentes du courant       : {len(exclues)}")
print(f"présentes mais toujours nulles en production : {len(vides_en_production)}")
if exclues:
    print("exemples d'exclusions :", exclues[:10])
print()

# Vue d'ensemble : part des features en dérive sur l'ensemble du contrat.
rapport_global = Report([DataDriftPreset(method="psi", columns=communes)])
resultat_global = rapport_global.run(
    current_data=fenetre_a[communes], reference_data=reference[communes]
).dict()

synthese = next(
    m["value"] for m in resultat_global["metrics"]
    if m.get("config", {}).get("type", "").endswith("DriftedColumnsCount")
)
print(f"features comparées : {len(communes)}")
print(f"features en dérive : {synthese['count']:.0f} ({synthese['share']:.1%})\n")

# Analyse détaillée sur les features suivies.
psi_baseline = psi_par_colonne(reference, fenetre_a, FEATURES_SUIVIES)
tableau = pd.DataFrame({
    "psi": psi_baseline.round(4),
    "bande": psi_baseline.map(bande),
    "importance": importances[psi_baseline.index].astype(int),
})
print(tableau)

# Dérive du taux de valeurs manquantes, traitée séparément.
manquantes = pd.DataFrame({
    "reference": reference[FEATURES_SUIVIES].isna().mean(),
    "production": fenetre_a[FEATURES_SUIVIES].isna().mean(),
})
manquantes["ecart"] = (manquantes["production"] - manquantes["reference"]).round(4)
print("\ntaux de valeurs manquantes :")
print(manquantes.sort_values("ecart", key=abs, ascending=False).head(10))

assert len(psi_baseline) == len(FEATURES_SUIVIES)
print("\nBaseline calculée.")
```

- [ ] **Étape 2 : Exécuter**

```bash
uv run python scratch/baseline.py
```

Attendu : la part de features en dérive est faible — le trafic simulé est tiré du même
jeu Home Credit que la référence. C'est le résultat honnête, et il doit être rapporté
tel quel : la section suivante démontrera la détection sur un cas réellement décalé.

- [ ] **Étape 3 : Reporter dans le notebook**

Cellule markdown d'introduction :

```markdown
## Dérive mesurée sur le trafic nominal

Comparaison de la fenêtre A au jeu d'entraînement. Le PSI est retenu plutôt que le test
de Kolmogorov-Smirnov : la p-value de ce dernier dépend de la taille d'échantillon, et à
20 000 observations de référence un écart sans portée métier ressort « significatif ».
Le PSI mesure une amplitude.

Bandes de lecture : moins de 0,10 stable, de 0,10 à 0,25 modérée, au-delà significative.
```

Cellules de code, dans cet ordre : le décompte des features exclues de la comparaison,
la vue d'ensemble, le tableau détaillé, les distributions comparées des trois features
au PSI le plus élevé (histogrammes superposés), puis le tableau des valeurs manquantes.

Ce dernier est précédé d'une courte cellule markdown expliquant pourquoi il est traité à
part : un primo-emprunteur sans historique de crédit est légitime ; mélanger ce phénomène
aux distributions produirait des dérives fantômes.

Cellule markdown de conclusion, à rédiger **après** avoir vu les chiffres : ce que le
résultat dit, sans le surinterpréter.

- [ ] **Étape 4 : Exécuter et committer**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_data_drift.ipynb
git add notebooks/07_data_drift.ipynb
git commit -m "feat(monitoring): mesurer la dérive du trafic nominal

Vue d'ensemble sur toutes les features communes, puis détail sur les 20
suivies. Le taux de valeurs manquantes est traité séparément : un dossier
sans historique de crédit est légitime, le mélanger aux distributions
produirait des dérives fantômes."
```

---

## Tâche 7 : Notebook § 5 — scénario contrôlé et balayage d'amplitude

**Fichiers :**
- Modifier : `notebooks/07_data_drift.ipynb`
- Créer puis supprimer : `scratch/scenario.py`

**Interfaces :**
- Consomme : `reference`, `fenetre_a`, `fenetre_b`, `FEATURES_SUIVIES`,
  `psi_par_colonne`, `bande`, `psi_baseline`
- Produit : `psi_scenario` (`pd.Series`) et `balayage` (`pd.DataFrame` à colonnes
  `decalage`, `psi_ext_source_2`, `part_derivee`), consommés par la tâche 10.

- [ ] **Étape 1 : Prototyper**

Créer `scratch/scenario.py` reprenant ce qui précède, puis :

```python
COLONNES_DECALEES = [c for c in FEATURES_SUIVIES if c.startswith("EXT_SOURCE")]

# La fenêtre B est réellement passée par l'API et stockée : elle prouve la chaîne
# complète API -> PostgreSQL -> analyse.
psi_scenario = psi_par_colonne(reference, fenetre_b, FEATURES_SUIVIES)
comparaison = pd.DataFrame({
    "psi_fenetre_a": psi_baseline.round(4),
    "psi_fenetre_b": psi_scenario[psi_baseline.index].round(4),
})
comparaison["bande_b"] = comparaison["psi_fenetre_b"].map(bande)
print(comparaison)

# Balayage d'amplitude, en mémoire : le décalage est une transformation déterministe,
# le faire transiter par HTTP n'apprendrait rien et peuplerait la base de fenêtres
# supplémentaires à démêler.
lignes = []
for decalage in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    simulee = fenetre_a.copy()
    for colonne in COLONNES_DECALEES:
        simulee[colonne] = (simulee[colonne] - decalage).clip(0, 1)
    psi = psi_par_colonne(reference, simulee, FEATURES_SUIVIES)
    lignes.append({
        "decalage": decalage,
        "psi_ext_source_2": round(psi.get("EXT_SOURCE_2", float("nan")), 4),
        "part_derivee": round((psi >= 0.10).mean(), 3),
    })

balayage = pd.DataFrame(lignes)
print("\nbalayage d'amplitude :")
print(balayage)

assert balayage.loc[balayage["decalage"] == 0.00, "psi_ext_source_2"].item() < 0.10
assert balayage["psi_ext_source_2"].is_monotonic_increasing
print("\nScénario calculé.")
```

- [ ] **Étape 2 : Exécuter**

```bash
uv run python scratch/scenario.py
```

Attendu : PSI croissant avec le décalage, nul à décalage 0. Noter le décalage à partir
duquel `psi_ext_source_2` franchit 0,10 puis 0,25 — ce sont les seuils de déclenchement
à citer en conclusion.

Si `is_monotonic_increasing` échoue, vérifier que `COLONNES_DECALEES` n'est pas vide.

- [ ] **Étape 3 : Reporter dans le notebook**

Cellule markdown :

```markdown
## Détection d'une dérive provoquée

La mesure précédente ne dit pas si le dispositif *saurait* voir une dérive. La fenêtre B
répond à cette question : son trafic a été envoyé à l'API avec les scores externes
décalés de 0,20, et il a suivi le même chemin que le trafic nominal — API, journalisation,
PostgreSQL, puis cette analyse.

Le balayage qui suit établit à partir de quelle amplitude la dérive devient détectable.
Il est calculé en mémoire sur la fenêtre A : le décalage est une transformation
déterministe, la faire transiter par HTTP n'apporterait aucune information.
```

Cellules de code : comparaison A/B, puis balayage, puis un graphique du PSI en fonction
du décalage avec les deux seuils tracés en lignes horizontales.

Cellule markdown de conclusion nommant les seuils de déclenchement observés.

- [ ] **Étape 4 : Exécuter et committer**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_data_drift.ipynb
git add notebooks/07_data_drift.ipynb
git commit -m "feat(monitoring): démontrer la détection sur une dérive provoquée

La fenêtre B a réellement traversé l'API et la base : elle valide la chaîne
complète. Le balayage d'amplitude, lui, est calculé en mémoire et établit le
seuil de décalage à partir duquel la dérive devient détectable."
```

---

## Tâche 8 : Notebook § 6 — dérive du score et de la décision

**Fichiers :**
- Modifier : `notebooks/07_data_drift.ipynb`
- Créer puis supprimer : `scratch/score.py`

**Interfaces :**
- Consomme : `fenetre_a`, `fenetre_b`
- Produit : `resume_decisions` — `pd.DataFrame` indexé par fenêtre, colonnes
  `predictions`, `taux_refus`, `proba_moyenne`, `proba_p95`. Consommé par la tâche 10.
- Produit aussi : `fenetres` — les deux fenêtres empilées avec une colonne `fenetre`
  valant `"A — nominal"` ou `"B — décalé"`. Ces deux libellés exacts servent d'index
  dans les tâches 9 et 10 : ne pas les modifier.

- [ ] **Étape 1 : Prototyper**

Créer `scratch/score.py` reprenant le chargement, puis :

```python
SEUIL_DECISION = 0.10  # seuil métier de la Partie 1 : coût = 10 x FN + 1 x FP

fenetres = pd.concat([
    fenetre_a.assign(fenetre="A — nominal"),
    fenetre_b.assign(fenetre="B — décalé"),
])

resume_decisions = fenetres.groupby("fenetre").apply(
    lambda g: pd.Series({
        "predictions": len(g),
        "taux_refus": (g["decision"] == "rejected").mean().round(4),
        "proba_moyenne": g["probability"].mean().round(4),
        "proba_p95": g["probability"].quantile(0.95).round(4),
    }),
    include_groups=False,
)
print(resume_decisions)

assert resume_decisions.loc["B — décalé", "taux_refus"] > resume_decisions.loc["A — nominal", "taux_refus"]
print("\nDérive du score confirmée.")
```

- [ ] **Étape 2 : Exécuter**

```bash
uv run python scratch/score.py
```

Attendu : le taux de refus de la fenêtre B dépasse nettement celui de la fenêtre A.
Valeurs **effectivement mesurées** sur les deux fenêtres générées en tâche 2 :
**20,7 %** de refus en fenêtre A (3 000 dossiers) contre **45,0 %** en fenêtre B
(1 000 dossiers, `--decalage 0.2`). Un facteur 2,2.

Ces chiffres diffèrent d'essais antérieurs (~13,5 % / ~39 %) menés sur des échantillons
de 200 et 120 dossiers : `simuler_trafic.py` tire avec `sample(n=nombre, random_state=42)`,
donc un `n` différent produit un échantillon différent. Les valeurs ci-dessus, assises
sur 4 000 dossiers, font foi.

Si `include_groups=False` provoque une erreur, la version de pandas est antérieure à
2.2 : retirer l'argument.

- [ ] **Étape 3 : Reporter dans le notebook**

Cellule markdown :

```markdown
## Dérive du score et de la décision

C'est la conséquence qui intéresse le métier : une dérive des entrées ne compte que si
elle déplace les décisions. Le seuil de décision est 0,10 et non 0,5, la métrique métier
pénalisant un mauvais client accepté dix fois plus qu'un bon client refusé.
```

Cellules de code : le tableau de résumé, puis les distributions de probabilité des deux
fenêtres superposées avec le seuil de décision tracé en vertical.

Cellule markdown de conclusion chiffrant l'écart de taux de refus.

- [ ] **Étape 4 : Exécuter et committer**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_data_drift.ipynb
git add notebooks/07_data_drift.ipynb
git commit -m "feat(monitoring): mesurer la dérive du score et du taux de refus

Une dérive des entrées ne compte que si elle déplace les décisions. Le taux
de refus est la métrique que le métier lit en premier."
```

---

## Tâche 9 : Notebook § 7 — métriques opérationnelles

**Fichiers :**
- Modifier : `notebooks/07_data_drift.ipynb`
- Créer puis supprimer : `scratch/operationnel.py`

**Interfaces :**
- Consomme : `fenetres` (tâche 8)
- Produit : `latences` — `pd.DataFrame` indexé par fenêtre, colonnes `p50`, `p95`,
  `p99`, `max`. Consommé par la tâche 10.

La mission demande explicitement ces métriques, au même titre que la dérive :
« distribution des scores prédits, latence de l'API, temps d'inférence » et
« détection d'anomalies (taux d'erreur, latence anormale) ».

- [ ] **Étape 1 : Prototyper**

Créer `scratch/operationnel.py` reprenant le chargement et `fenetres`, puis :

```python
latences = fenetres.groupby("fenetre")["latency_ms"].describe(
    percentiles=[0.5, 0.95, 0.99]
)[["50%", "95%", "99%", "max"]].round(3)
latences.columns = ["p50", "p95", "p99", "max"]
print("latence d'inférence (ms) :")
print(latences)

# Anomalies : une latence isolée très au-dessus du p99 signale une contention.
seuil_anomalie = fenetres["latency_ms"].quantile(0.99) * 3
anomalies = fenetres[fenetres["latency_ms"] > seuil_anomalie]
print(f"\nseuil d'anomalie (3 x p99) : {seuil_anomalie:.2f} ms")
print(f"prédictions au-dessus : {len(anomalies)} sur {len(fenetres)}")

# Couverture des dossiers reçus : une baisse expliquerait une dérive des décisions
# sans qu'aucune distribution n'ait bougé.
couverture = fenetres.groupby("fenetre")[["application_ratio", "history_ratio"]].mean().round(4)
print("\ncouverture moyenne des dossiers :")
print(couverture)

assert latences["p50"].max() < 1000, "latence médiane anormalement élevée"
print("\nMétriques opérationnelles calculées.")
```

- [ ] **Étape 2 : Exécuter**

```bash
uv run python scratch/operationnel.py
```

Attendu : latences médianes de l'ordre du dixième de milliseconde à quelques
millisecondes (mesures antérieures : 0,3 à 0,5 ms de moyenne).

- [ ] **Étape 3 : Reporter dans le notebook**

Cellule markdown :

```markdown
## Métriques opérationnelles

La dérive des données n'est qu'une des façons dont un service se dégrade. Latence,
anomalies et complétude des dossiers reçus complètent le tableau.

La couverture mérite une attention particulière : si les appelants se mettent à envoyer
des dossiers moins renseignés, les décisions changent sans qu'aucune distribution n'ait
bougé. Confondre les deux causes conduirait à réentraîner un modèle qui n'a rien.
```

Cellules de code : tableau des latences, détection d'anomalies, tableau de couverture,
et un histogramme des latences en échelle logarithmique.

Ajouter une cellule markdown signalant l'angle mort : les requêtes refusées en 422 ne
sont pas journalisées, le taux d'erreur applicatif n'est donc pas mesurable ici.

- [ ] **Étape 4 : Exécuter et committer**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_data_drift.ipynb
git add notebooks/07_data_drift.ipynb
git commit -m "feat(monitoring): rapporter les métriques opérationnelles du service

Latence p50/p95/p99, anomalies et couverture des dossiers reçus. La mission
les demande au même titre que la dérive. La couverture distingue une dérive
du modèle d'une dégradation de la qualité des dossiers reçus."
```

---

## Tâche 10 : Notebook § 8 — conclusions et points de vigilance

**Fichiers :**
- Modifier : `notebooks/07_data_drift.ipynb`

**Interfaces :**
- Consomme : `psi_baseline`, `psi_scenario`, `balayage`, `resume_decisions`, `latences`
- Produit : rien — section terminale.

La mission demande « une présentation de l'étude sur la dérive des données et les points
de vigilance résultants ». Cette section est donc un livrable en soi, pas une politesse
de fin de notebook.

- [ ] **Étape 1 : Rédiger le tableau de synthèse**

Une cellule de code assemblant les résultats **déjà calculés**, sans recalcul :

```python
seuil_detection = balayage.loc[balayage["psi_ext_source_2"] >= 0.10, "decalage"]
seuil_detection = seuil_detection.min() if len(seuil_detection) else float("nan")

synthese_finale = pd.DataFrame({
    "fenêtre A — nominal": {
        "PSI maximal (features suivies)": round(psi_baseline.max(), 4),
        "features en dérive (sur 20)": int((psi_baseline >= 0.10).sum()),
        "taux de refus": resume_decisions.loc["A — nominal", "taux_refus"],
        "latence p95 (ms)": latences.loc["A — nominal", "p95"],
    },
    "fenêtre B — décalé": {
        "PSI maximal (features suivies)": round(psi_scenario.max(), 4),
        "features en dérive (sur 20)": int((psi_scenario >= 0.10).sum()),
        "taux de refus": resume_decisions.loc["B — décalé", "taux_refus"],
        "latence p95 (ms)": latences.loc["B — décalé", "p95"],
    },
})
print(f"décalage minimal détecté : {seuil_detection}")
synthese_finale
```

- [ ] **Étape 2 : Rédiger les points de vigilance**

Une cellule markdown structurée par les quatre questions auxquelles un dispositif de
surveillance doit répondre. Chaque point doit être **chiffré à partir des résultats
obtenus**, pas générique :

1. **Quoi surveiller** — les features suivies, le taux de refus, la couverture des
   dossiers, la latence p95.
2. **À quelle fréquence** — à justifier par le volume observé et la vitesse à laquelle
   une dérive devient détectable dans le balayage.
3. **À partir de quel seuil agir** — reprendre les bandes PSI et le seuil de
   déclenchement mesuré en tâche 7.
4. **Quand réentraîner** — critère explicite, distinguant une dérive des entrées d'une
   dégradation de performance réelle.

Ajouter un paragraphe sur ce que cette analyse **ne** couvre pas : la dérive de
performance du modèle (*model drift*) est hors d'atteinte, puisqu'elle exigerait de
connaître le défaut réel des clients scorés, information qui n'arrive que des mois plus
tard. Le dire vaut mieux que laisser croire le sujet traité.

- [ ] **Étape 3 : Relire le notebook entier pour la sobriété**

Parcourir toutes les cellules markdown et supprimer : les formules d'accompagnement, les
emojis, les répétitions d'explications déjà données, les commentaires de code qui
paraphrasent le code. Chaque cellule markdown doit dire soit ce qu'on va regarder, soit
ce que le résultat montre.

- [ ] **Étape 4 : Exécuter le notebook de bout en bout**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_data_drift.ipynb
```

Attendu : exécution complète sans erreur, sorties sauvegardées dans le fichier.

- [ ] **Étape 5 : Committer**

```bash
git add notebooks/07_data_drift.ipynb
git commit -m "feat(monitoring): conclusions et points de vigilance

Quoi surveiller, à quelle fréquence, à partir de quel seuil agir, quand
réentraîner — chiffré sur les résultats obtenus. Le model drift est
explicitement déclaré hors d'atteinte : il exigerait de connaître le défaut
réel des clients, qui n'arrive que des mois plus tard."
```

---

## Tâche 11 : Documentation et nettoyage

**Fichiers :**
- Modifier : `README.md`
- Modifier : `.gitignore`
- Supprimer : `scratch/`

- [ ] **Étape 1 : Protéger contre un export de rapport**

Dans `.gitignore`, sous la section « Données de production / monitoring » :

```
# Rapports Evidently exportés : ils agrègent des distributions de dossiers
# Home Credit et n'ont pas leur place dans un dépôt public.
monitoring/*.html
```

- [ ] **Étape 2 : Corriger la description de `monitoring/` dans le README**

Le README décrit aujourd'hui `monitoring/` comme accueillant « logs de production et
rapports de drift (étape 3) ». Le stockage étant en base et l'analyse dans le notebook,
ce répertoire reste vide. Remplacer la ligne par :

```
├── monitoring/           # réservé aux exports locaux, non versionnés
```

- [ ] **Étape 3 : Renvoyer au notebook depuis la section monitoring du README**

Ajouter à la fin de la section « Interpréter le monitoring » :

```markdown
L'analyse de la dérive des données est dans
[`notebooks/07_data_drift.ipynb`](notebooks/07_data_drift.ipynb) : comparaison du trafic
de production au jeu d'entraînement, démonstration de la détection sur une dérive
provoquée, métriques opérationnelles et points de vigilance.
```

- [ ] **Étape 4 : Mettre à jour le tableau d'avancement du README**

Remplacer la ligne de l'étape 3 par :

```
| 3 | Stockage des données de production + analyse du data drift | ✅ en place |
```

- [ ] **Étape 5 : Supprimer les prototypes**

```bash
rm -rf scratch/
git status --porcelain
```

Attendu : `scratch/` n'apparaît pas, et aucun fichier de données n'est en attente.

- [ ] **Étape 6 : Vérification finale**

```bash
uv run ruff check api tests src scripts
uv run pytest -q
git status --porcelain
```

Attendu : ruff passe, la suite de tests reste verte, et aucun fichier de données client
n'est suivi.

- [ ] **Étape 7 : Committer**

```bash
git add README.md .gitignore
git commit -m "docs: renvoyer au notebook de drift et corriger la description de monitoring/

Le README promettait des rapports de drift dans monitoring/ : le stockage est
en base et l'analyse dans le notebook, ce répertoire reste vide. Ajout d'une
règle .gitignore contre un export HTML de rapport Evidently, qui agrégerait
des distributions de dossiers Home Credit dans un dépôt public."
```

---

## Critères d'acceptation

Repris de la spec, § 9 :

1. Le notebook s'exécute de bout en bout, sorties comprises, sur la pile locale.
2. Les deux cas de contrôle de l'instrument passent (tâche 4).
3. Les huit sections sont présentes et concluent chacune par une lecture, pas seulement
   par un graphique.
4. Les points de vigilance sont actionnables et chiffrés : quoi surveiller, à quelle
   fréquence, à partir de quel seuil agir.
5. Aucune donnée client n'est versionnée.
6. `ruff check` passe et la suite de tests existante reste verte.
7. Le README renvoie au notebook et sa description de `monitoring/` est corrigée.
