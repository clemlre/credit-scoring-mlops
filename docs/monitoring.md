# Suivi en production : journalisation et stockage des prédictions

Ce document décrit **ce que l'API enregistre**, **où**, **pourquoi ce choix**, et
comment relire ces données. C'est la matière première de l'analyse de dérive
(étape suivante) et de toute enquête sur une décision contestée.

## Ce qui est enregistré

À chaque prédiction rendue — et uniquement à celles-là, une requête refusée en 422
n'ayant produit aucun score à surveiller :

| Champ | Pourquoi il est là |
|---|---|
| `request_id` | Renvoyé au client dans l'en-tête `X-Request-ID`. C'est la clé qui relie une réclamation à la ligne exacte en base. |
| `occurred_at` | Horodatage UTC. Toute analyse de dérive est une comparaison de fenêtres temporelles. |
| `endpoint` | `/predict` ou `/predict/batch` — les usages n'ont ni le même profil ni la même criticité. |
| `model_version` | Sans elle, impossible de distinguer une dérive des données d'un changement de modèle. |
| `threshold` | Le seuil **appliqué ce jour-là**. S'il est un jour réajusté, l'historique reste interprétable. |
| `probability`, `decision` | Le résultat lui-même. |
| `features_provided`, `features_missing`, `application_ratio`, `history_ratio` | La **couverture** du dossier. Un taux de refus qui monte peut venir du modèle… ou d'appelants qui envoient des dossiers plus incomplets. Sans cette colonne, les deux causes sont indiscernables. |
| `latency_ms` | Temps d'inférence. Base de référence pour l'étape d'optimisation. |
| `features` | Le payload **tel que reçu**, en `JSONB`. C'est ce qui rend la dérive mesurable. |

## Deux canaux, et pourquoi ils sont distincts

```
                          ┌──────────────────────────────────────┐
   POST /predict ────────▶│  API (réponse renvoyée immédiatement) │
                          └──────────────┬───────────────────────┘
                                         │ BackgroundTasks (après la réponse)
                          ┌──────────────┴───────────────┐
                          ▼                              ▼
             stdout, JSON par ligne            PostgreSQL (table predictions)
             ─────────────────────             ──────────────────────────────
             toujours actif                    si DATABASE_URL est défini
             SANS valeur de feature            AVEC les features, en JSONB
             exploitation / incidents          monitoring / dérive / audit
```

**Pourquoi deux canaux et pas un seul ?**

- La sortie standard est le transport de journaux natif d'un conteneur : Docker,
  Kubernetes et Hugging Face Spaces la collectent sans rien configurer. Elle reste
  disponible même si la base est tombée — donc **aucune prédiction n'est jamais
  totalement perdue**.
- Mais elle n'est pas interrogeable. Calculer « la distribution de `EXT_SOURCE_2`
  sur les 7 derniers jours » sur des fichiers de journaux est un travail d'ETL.
  En SQL, c'est une ligne.
- Et surtout : **les journaux applicatifs finissent souvent chez un tiers**
  (Datadog, CloudWatch, l'hébergeur). Le revenu, l'âge et l'historique de crédit
  d'un demandeur n'ont rien à y faire. Les features ne partent donc **que** dans la
  base que « Prêt à Dépenser » contrôle.

## Pourquoi PostgreSQL

C'est le moteur déjà exploité en production par l'équipe : le choix n'ajoute aucune
compétence ni aucune astreinte nouvelle à maintenir.

Trois autres raisons, techniques :

1. **`JSONB` résout le problème du schéma.** Le modèle a 779 features. Une table à
   779 colonnes serait ingérable et, surtout, **cassée le jour où le modèle change
   de contrat** — or comparer deux versions de modèle est exactement ce que le
   monitoring doit permettre. Le `JSONB` absorbe le changement.
2. **Les features restent interrogeables**, contrairement à un blob : voir les
   requêtes ci-dessous.
3. **L'écriture est concurrente**, ce que ni un fichier JSON Lines ni SQLite ne
   garantissent dès qu'on met deux instances de l'API derrière un répartiteur.

**Ce qui a été écarté :**

| Option | Pourquoi non |
|---|---|
| Fichier JSON Lines sur volume | Simple, mais non requêtable et non concurrent. Il faudrait tout relire pour la moindre agrégation. |
| SQLite | Requêtable, mais mono-écrivain : le jour où l'API est répliquée, il devient le goulot. |
| Elasticsearch | Excellent pour les journaux, mais un service de plus à opérer pour un besoin que PostgreSQL couvre. |

**Ce que ce choix coûte** : une agrégation sur une clé `JSONB` est plus lente que sur
une vraie colonne. À l'échelle mesurée (voir plus bas) c'est sans effet ; passé
plusieurs dizaines de millions de lignes, on promouvrait les features les plus
consultées en colonnes générées, ou l'on basculerait la table en partitionnement
mensuel.

## Volumétrie mesurée

Relevé sur la pile locale, après 336 prédictions réelles :

```
 taille_table | lignes
--------------+--------
 3200 kB      |    336
```

Soit **≈ 9,5 ko par prédiction**, index compris, pour un dossier complet
(245 features renseignées). À 1 000 prédictions par jour : ~9,5 Mo/jour,
**~3,5 Go/an**. Une rétention glissante de 12 à 24 mois est donc tenable sur une
petite instance ; au-delà, on archiverait les partitions anciennes.

## Ce que le service fait quand la base tombe

Décision structurante : **une panne de monitoring n'est pas une panne de
production.**

- L'écriture a lieu dans une tâche d'arrière-plan, *après* l'envoi de la réponse :
  le temps de réponse ne dépend pas de la base.
- Toute exception y est absorbée à deux niveaux (`PredictionLog.record`, puis
  `_journaliser_sans_echec` dans `api/main.py`).
- `GET /health` **reste à 200** et signale l'état dans `prediction_log.database` :

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1",
  "prediction_log": { "stdout": true, "database": "unavailable", "last_error": "..." }
}
```

Renvoyer 503 parce que la base de monitoring est indisponible ferait retirer l'API
du trafic par le répartiteur de charge : une panne d'observabilité deviendrait une
panne de service. C'est l'inverse de ce qu'on veut.

Les trois états possibles : `disabled` (aucune base configurée — normal en test et
en démonstration), `ready`, `unavailable`.

## Lancer la pile localement

```bash
cp .env.example .env          # facultatif : pour changer les ports ou le mot de passe
docker compose up -d --build  # API + PostgreSQL
curl http://localhost:8000/health
```

Alimenter le journal avec du trafic réaliste :

```bash
# Dossiers réels tirés du jeu de la Partie 1 (s'il est présent sur le poste)
python scripts/simuler_trafic.py --nombre 200

# Trafic volontairement décalé, pour vérifier qu'un détecteur de dérive se déclenche
python scripts/simuler_trafic.py --nombre 120 --decalage 0.2
```

> Le port hôte de l'API est surchargeable (`API_PORT=8001 docker compose up -d`) :
> 8000 est souvent déjà pris sur un poste de développement.

## Relire les données

En ligne de commande :

```bash
docker exec scoring-db psql -U scoring -d monitoring
```

**Répartition des décisions et latence moyenne**

```sql
SELECT decision,
       count(*),
       round(avg(probability)::numeric, 4) AS proba_moy,
       round(avg(latency_ms)::numeric, 2)  AS latence_moy_ms
FROM predictions
GROUP BY decision;
```

```
 decision | count | proba_moy | latence_moy_ms
----------+-------+-----------+----------------
 accepted |   261 |    0.0391 |           0.47
 rejected |    75 |    0.2009 |           0.30
```

**Taux de refus par tranche d'un score externe** — l'intérêt du `JSONB` : la feature
est agrégée directement, sans table dédiée.

```sql
SELECT width_bucket((features->>'EXT_SOURCE_2')::float, 0, 1, 5) AS tranche,
       count(*)                                                  AS predictions,
       round(100.0 * count(*) FILTER (WHERE decision = 'rejected') / count(*), 1)
                                                                 AS taux_refus_pct
FROM predictions
WHERE features ? 'EXT_SOURCE_2'
GROUP BY 1 ORDER BY 1;
```

```
 tranche | predictions | taux_refus_pct
---------+-------------+----------------
       1 |          44 |           63.6
       2 |          67 |           32.8
       3 |         136 |           13.2
       4 |          88 |            8.0
       5 |           1 |            0.0
```

La relation est monotone, comme attendu : plus le score externe est bas, plus le
modèle refuse. Une inversion de cette courbe serait un signal d'alerte fort.

**Retrouver une décision contestée**

```sql
SELECT occurred_at, probability, decision, threshold, model_version, features
FROM predictions
WHERE request_id = '0eab2763-b69c-4e49-9301-ffcea0411d00';
```

## Console graphique et captures d'écran

Pour le livrable « captures d'écran de la solution de stockage » :

```bash
docker compose --profile outils up -d      # ajoute pgAdmin
```

Ouvrir <http://localhost:5050>. La connexion **« Monitoring - scoring de credit »**
est déjà déclarée (voir `docs/pgadmin/servers.json`) ; il reste à saisir le mot de
passe de développement (`scoring_dev` par défaut, ou celui de votre `.env`).

Captures utiles :

1. l'arborescence `monitoring → Schemas → public → Tables → predictions` ;
2. la structure de la table (colonnes et types, dont `features` en `jsonb`) ;
3. les lignes réelles (`View/Edit Data → All Rows`), en dépliant une cellule
   `features` ;
4. le résultat d'une des requêtes d'agrégation ci-dessus dans le *Query Tool* ;
5. `docker compose ps` montrant les conteneurs et le volume `pgdata`.

Ranger les images dans `docs/screenshots/`.

## Ce qui n'est pas couvert (et pourquoi c'est assumé)

- **Les requêtes refusées (422) ne sont pas journalisées.** Un taux de rejet en
  hausse serait pourtant un signal utile — c'est la première extension à prévoir.
- **Aucune purge automatique.** La rétention devra être décidée avec le métier
  (obligation de conservation d'une décision de crédit) puis appliquée par une
  tâche planifiée ou un partitionnement.
- **Une tâche d'arrière-plan par requête**, sans file bornée. Suffisant ici ; sous
  forte charge, il faudrait une file interne à consommateur unique, ou un envoi
  vers un collecteur externe.
- **Le `request_id` est généré par l'API.** Dans un système distribué, on
  reprendrait plutôt un identifiant de corrélation transmis par l'appelant
  (`traceparent`).

## Tests

| Niveau | Fichier | Ce qui est garanti |
|---|---|---|
| Unitaire | `tests/test_storage.py` | Canaux alimentés, erreurs absorbées, schéma créé une seule fois, chaîne de connexion jamais divulguée. Avec un faux pool : **le SQL n'y est pas validé**. |
| Unitaire (HTTP) | `tests/test_api.py::TestJournalisationDesPredictions` | En-tête `X-Request-ID`, journalisation effective, 422 non journalisée, panne du journal sans effet sur la réponse. |
| Intégration | `tests/test_storage.py::TestIntegrationPostgres` | SQL réellement exécutable, index présents, `JSONB` relisible et agrégeable. |

Les tests d'intégration sont ignorés sans `DATABASE_URL`. La CI en fournit un
(service PostgreSQL éphémère) : ils **s'exécutent donc à chaque pipeline**.

```bash
# En local, avec la pile démarrée :
DATABASE_URL="postgresql://scoring:scoring_dev@127.0.0.1:5432/monitoring" \
  uv run pytest --cov=api
```
