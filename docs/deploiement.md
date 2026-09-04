# Déploiement de l'API

Le pipeline CI/CD déploie l'API automatiquement à chaque poussée sur `main`, **à
condition** que les identifiants du service cible soient configurés. Ce document décrit
la configuration à faire une fois, et ce qui se passe ensuite.

## Ce que fait le pipeline

```
push sur main
   │
   ├─ 1. Lint et tests ......... ruff + pytest, plancher de couverture 95 %
   │
   ├─ 2. Image Docker .......... build, démarrage du conteneur, test de fumée
   │                             puis publication sur le registre GHCR
   │
   └─ 3. Déploiement ........... poussée vers Hugging Face Spaces
                                 puis test de fumée sur le service déployé
```

Une étape échoue ⇒ les suivantes ne s'exécutent pas. Une image n'est publiée que si le
conteneur a réellement répondu, et un déploiement n'est considéré comme réussi que si le
service déployé répond correctement.

## Deux cibles de déploiement

### Registre d'images GHCR — actif sans configuration

À chaque poussée sur `main`, l'image est publiée sur
`ghcr.io/clemlre/credit-scoring-mlops`, étiquetée `latest` et par empreinte de commit.
Aucun secret à créer : le pipeline utilise le jeton éphémère fourni par GitHub.

```bash
docker run -p 8000:8000 ghcr.io/clemlre/credit-scoring-mlops:latest
```

### Hugging Face Spaces — service en ligne, à configurer une fois

C'est la cible recommandée par l'énoncé du projet. Le déploiement reste **inactif tant
que les identifiants ne sont pas fournis** : le job le signale et se termine sans échouer,
pour ne pas faire échouer une CI par ailleurs valide.

#### Configuration (une seule fois)

1. **Créer le Space** sur <https://huggingface.co/new-space>
   - propriétaire : votre compte ;
   - nom : `credit-scoring-api` (par exemple) ;
   - **SDK : Docker** — surtout pas Gradio ou Streamlit, l'image est fournie par le
     dépôt ;
   - visibilité : publique.

2. **Créer un jeton d'accès** sur
   <https://huggingface.co/settings/tokens> — type **Write**.

3. **Déclarer les identifiants dans GitHub**, dans
   *Settings → Secrets and variables → Actions* :

   | Type | Nom | Valeur |
   |---|---|---|
   | **Secret** | `HF_TOKEN` | le jeton créé à l'étape 2 |
   | **Variable** | `HF_SPACE` | `votre-compte/credit-scoring-api` |

   Le jeton est un **secret**, jamais une variable : GitHub masque sa valeur dans les
   journaux et interdit sa lecture après enregistrement. Il n'apparaît à aucun moment
   dans un fichier du dépôt.

4. **Relancer le pipeline** — n'importe quelle poussée sur `main`, ou
   *Actions → CI/CD → Run workflow*.

#### Ce que le pipeline fait alors

- Il génère l'en-tête de configuration attendu par Hugging Face (`sdk: docker`,
  `app_port: 8000`) en tête du README, **sans le committer dans le dépôt GitHub** — le
  README du projet reste lisible.
- Il pousse le dépôt vers le Space, qui reconstruit l'image à partir du `Dockerfile`.
- Il **interroge l'API Hugging Face** jusqu'à ce que le Space passe à l'état `RUNNING`
  (15 min au maximum), puis lance `scripts/smoke_test.py` sur l'URL publique que
  l'API déclare. Un build en échec (`BUILD_ERROR`, `RUNTIME_ERROR`, `CONFIG_ERROR`)
  fait échouer le pipeline immédiatement, sans attendre la fin du délai.

L'API est ensuite accessible sur `https://<compte>-<nom-du-space>.hf.space`, avec sa
documentation Swagger sur `/docs`.

## Vérifier un déploiement à la main

```bash
python scripts/smoke_test.py https://votre-compte-credit-scoring-api.hf.space
```

Le script contrôle la disponibilité, la version du modèle servi, la cohérence de la
décision avec le seuil, et le refus d'une entrée invalide.

## Pièges rencontrés

| Symptôme | Cause | Correctif appliqué |
|---|---|---|
| Le Space se construit puis affiche « no healthy upstream » | `app_port` ne correspond pas au port exposé | `app_port: 8000`, aligné sur le `Dockerfile` |
| `import lightgbm` échoue au démarrage du conteneur | `libgomp1` absent des images `slim` | installé explicitement dans le `Dockerfile` |
| Le Space ignore la configuration | en-tête YAML absent du README | généré par le pipeline avant la poussée |
| `! [remote rejected] … shallow update not allowed` | `actions/checkout` fait un clone superficiel, impossible à pousser vers un autre serveur Git | `fetch-depth: 0` sur le job de déploiement |
| Le test de fumée échoue alors que le Space finit par tourner | le premier build d'une image de 583 Mo dépasse largement le délai d'attente fixe qui était utilisé | attente de l'état `RUNNING` via l'API, au lieu d'une durée en dur |
