# Suivi en production : journalisation et stockage

Ce document décrit **ce que l'API enregistre**, **où**, **pourquoi ce choix**, et
comment relire ces données. C'est la matière première de l'analyse de dérive
(étape suivante) et de toute enquête sur une décision contestée.

## Ce qui est enregistré

Deux tables, reliées par `request_id`.

### `predictions` — une ligne par dossier scoré

| Champ | Pourquoi il est là |
|---|---|
| `request_id` | Renvoyé au client dans l'en-tête `X-Request-ID`. C'est la clé qui relie une réclamation à la ligne exacte en base. |
| `occurred_at` | Horodatage UTC. Toute analyse de dérive est une comparaison de fenêtres temporelles. |
| `endpoint` | `/predict` ou `/predict/batch` — les usages n'ont ni le même profil ni la même criticité. |
| `model_version` | Sans elle, impossible de distinguer une dérive des données d'un changement de modèle. |
| `threshold` | Le seuil **appliqué ce jour-là**. S'il est un jour réajusté, l'historique reste interprétable. |
| `probability`, `decision` | Le résultat lui-même. |
| `features_provided`, `features_missing`, `application_ratio`, `history_ratio` | La **couverture** du dossier. Un taux de refus qui monte peut venir du modèle… ou d'appelants qui envoient des dossiers plus incomplets. Sans cette colonne, les deux causes sont indiscernables. |
| `latency_ms` | Temps d'**inférence** seul (construction de la matrice + modèle), hors validation et HTTP. Pour un lot, durée de l'appel divisée par le nombre de dossiers. |
| `features` | Le payload **tel que reçu**, en `JSONB`. C'est ce qui rend la dérive mesurable. |

### `requests` — une ligne par appel HTTP, erreurs comprises

| Champ | Pourquoi il est là |
|---|---|
| `request_id` | Même identifiant que dans `predictions` et dans l'en-tête `X-Request-ID`. |
| `occurred_at`, `method`, `path` | Quand, et sur quelle route. |
| `status_code` | 200, 422 (entrée refusée), 500 (erreur interne), 503 (modèle absent). C'est ce qui donne le **taux d'erreur**. |
| `duration_ms` | Temps de traitement **de la requête** côté serveur, mesuré par un middleware : validation + inférence + sérialisation. C'est la latence que perçoit l'appelant, hors réseau. |

Routes suivies : `/predict`, `/predict/batch`, `/model/info`, `/features`. `/health` en
est exclue : la sonde du conteneur l'appelle toutes les 30 s.

Séparer les deux tables évite de mélanger deux grains : un appel `/predict/batch` de
200 dossiers donne **une** ligne `requests` et **200** lignes `predictions`.

## Deux canaux, et pourquoi ils sont distincts

```
                          ┌──────────────────────────────────────┐
   POST /predict ────────▶│  API (réponse renvoyée immédiatement) │
                          └──────────────┬───────────────────────┘
                                         │ middleware, après la réponse    
                          ┌──────────────┴───────────────┐
                          ▼                              ▼
             stdout, JSON par ligne            PostgreSQL (predictions, requests)
             ─────────────────────             ──────────────────────────────────
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

- L'écriture a lieu *après* l'envoi de la réponse, dans le middleware
  (`api/tracking.py`), en une transaction pour la requête et ses prédictions : le temps
  de réponse ne dépend pas de la base.
- Toute exception y est absorbée à deux niveaux (`PredictionLog.record_request`, puis
  `_write_log` dans `api/tracking.py`).
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

**Taux d'erreur et latence par route** (table `requests`)

```sql
SELECT path,
       count(*)                                                        AS appels,
       round(100.0 * count(*) FILTER (WHERE status_code >= 400) / count(*), 2)
                                                                       AS taux_erreur_pct,
       round(percentile_cont(0.5)  WITHIN GROUP (ORDER BY duration_ms)::numeric, 2) AS p50_ms,
       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)::numeric, 2) AS p95_ms
FROM requests
WHERE occurred_at > now() - interval '1 day'
GROUP BY path ORDER BY path;
```

Un taux d'erreur à 500 est un incident ; un taux de 422 qui monte signale plutôt un
appelant qui a changé son format d'envoi.

**Part de l'inférence dans le temps de requête** — la jointure sur `request_id` :

```sql
SELECT r.path,
       round(avg(r.duration_ms)::numeric, 2)   AS requete_ms,
       round(avg(p.latency_ms)::numeric, 2)    AS inference_ms
FROM requests r
JOIN (SELECT request_id, sum(latency_ms) AS latency_ms
      FROM predictions GROUP BY request_id) p USING (request_id)
GROUP BY r.path;
```

**Retrouver une décision contestée**

```sql
SELECT occurred_at, probability, decision, threshold, model_version, features
FROM predictions
WHERE request_id = '0eab2763-b69c-4e49-9301-ffcea0411d00';
```

## Tableau de bord

`monitoring/dashboard.py` est une application Streamlit branchée directement sur la base.

```bash
uv sync --group monitoring
uv run --group monitoring streamlit run monitoring/dashboard.py
# DATABASE_URL pour viser une autre base que celle de docker-compose.yml
```

Elle n'écoute que sur `127.0.0.1` (`.streamlit/config.toml`) : elle affiche des données
clients et n'a pas à être joignable depuis le réseau.

**Ce qu'elle montre**, sur une fenêtre choisie (dernière heure, 24 h, 7 jours…) :

| Zone | Contenu | Source |
|---|---|---|
| Alertes | erreurs 5xx, taux de 4xx > 5 %, latence p95 > 100 ms, dérive significative | calculées par `monitoring/indicateurs.py` |
| Cartes | appels, taux d'erreur, latence p95 de `/predict`, prédictions, taux de refus, nombre de features en dérive | `requests`, `predictions` |
| Activité et erreurs | appels par minute (ou heure) colorés par classe de statut, détail des erreurs par route | `requests` |
| Scores et décisions | distribution des probabilités avec le seuil de 0,10, taux de refus dans le temps | `predictions` |
| Latence | p50 et p95 de `POST /predict` dans le temps, médiane de l'inférence seule | `requests`, `predictions` |
| Dérive des données | PSI des 20 features suivies, taux de manquants référence contre production | `predictions.features` |

**La référence de dérive** est `monitoring/profil_reference.json` : les déciles des 20
features au plus fort gain, calculés sur les mêmes 20 000 dossiers d'entraînement que le
notebook. Il ne contient aucune ligne client, seulement des bornes et des proportions,
et se reconstruit avec `monitoring/construire_reference.py` quand le modèle change. Le PSI
est calculé par déciles de la référence : les valeurs sont du même ordre que celles
d'Evidently dans le notebook, sans être identiques (le découpage des classes diffère).

Les seuils d'alerte sont des constantes de `monitoring/indicateurs.py`, testées dans
`tests/test_indicateurs.py`.

**Analyser une période précise** : ajouter `?du=…&au=…` à l'URL, en ISO 8601 (UTC par
défaut), par exemple `http://127.0.0.1:8501/?du=2026-09-10T15:59Z&au=2026-09-10T16:04Z`.
C'est ce qui sert à relire un incident passé, ou à envoyer à quelqu'un le lien exact de
la fenêtre à regarder.

### Captures du tableau de bord

Scénario rejoué le 10 septembre contre l'API optimisée (2 workers), dans une base dédiée
`monitoring_demo` : 1 500 dossiers réels en lots, puis 7 minutes de trafic unitaire
nominal, puis 4 minutes de trafic dont les scores externes sont décalés de 0,2 ; 2 % des
dossiers sont volontairement invalides tout du long.

| Fichier | Ce qu'il montre |
|---|---|
| `dashboard-1-activite.png` | 2 633 appels par minute, 2,5 % de 422 (les dossiers invalides), aucune 500 |
| `dashboard-2-scores.png` | la distribution des scores et le taux de refus, qui passe d'environ 20 % à 47 % au début du trafic décalé |
| `dashboard-3-latence.png` | `POST /predict` stable autour de 4 ms en p50 et 7 à 8 ms en p95, inférence seule 0,8 ms |
| `dashboard-4-derive.png` | sur tout le scénario : seule `PAYMENT_RATE` dérive franchement (déjà relevé dans le notebook), les scores externes restent sous 0,25 car le trafic décalé n'en est qu'un cinquième |
| `dashboard-5-derive-trafic-decale.png` | sur la seule phase décalée : les trois `EXT_SOURCE` en dérive significative (PSI 3,0, 2,3 et 1,4), taux de refus 45,6 % |

La capture 5 recoupe la fenêtre B du notebook (PSI 3,04, 2,17 et 1,69, refus 45,0 %)
avec un autre calcul de PSI et d'autres dossiers : les deux instruments disent la même
chose.

## Console graphique et captures d'écran

Pour le livrable « captures d'écran de la solution de stockage » :

```bash
docker compose --profile outils up -d      # ajoute pgAdmin
```

Ouvrir <http://localhost:5050>. La connexion **« Monitoring - scoring de credit »**
est déjà déclarée (voir `docs/pgadmin/servers.json`) ; il reste à saisir le mot de
passe de développement (`scoring_dev` par défaut, ou celui de votre `.env`).

### Captures produites

Elles sont dans `docs/screenshots/` et constituent le livrable « captures d'écran de la
solution de stockage ».

| Fichier | Ce qu'il montre |
|---|---|
| `stockage-1-arborescence.png` | l'arborescence `monitoring → Schemas → public → Tables → predictions`, dépliée jusqu'aux colonnes et contraintes |
| `stockage-2-structure-table.png` | les 14 colonnes de la table et leurs types, `features` compris |
| `stockage-3-lignes-reelles.png` | des prédictions réellement journalisées, avec deux valeurs extraites du `jsonb` (`EXT_SOURCE_2`, `AMT_CREDIT`) et `jsonb_typeof` |
| `stockage-4-agregation-suivi.png` | volume, taux de refus, latence et couverture agrégés par minute — on y retrouve les deux fenêtres analysées dans le notebook : 3 000 prédictions à 20,70 % de refus, puis 1 000 à 45,00 % |
| `stockage-5-infrastructure.png` | les trois conteneurs, le volume `pgdata` et son point de montage, le volume de lignes et la taille de la table |

La capture 4 est la plus utile en soutenance : elle montre la même dérive du taux de
refus que le notebook, mais lue directement en base, sans Python ni Evidently.

**À savoir sur le contenu de la table.** Elle mélange le trafic simulé du 28 août
(les deux fenêtres d'analyse) et des prédictions plus récentes issues des tests
d'intégration, qui écrivent dans la même base en développement. L'analyse de dérive n'en
est pas affectée : le notebook borne ses fenêtres sur des horodatages explicites plutôt
que de prendre la table entière. En production, les environnements seraient séparés.

## Ce qui n'est pas couvert (et pourquoi c'est assumé)

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
| Unitaire (HTTP) | `tests/test_api.py::TestJournalisationDesPredictions` | En-tête `X-Request-ID`, prédictions journalisées, chaque appel journalisé avec son statut (200, 422, 500), `/health` exclue, panne du journal sans effet sur la réponse. |
| Intégration | `tests/test_storage.py::TestIntegrationPostgres` | SQL réellement exécutable pour les deux tables, index présents, `JSONB` relisible et agrégeable. |

Les tests d'intégration sont ignorés sans `DATABASE_URL`. La CI en fournit un
(service PostgreSQL éphémère) : ils **s'exécutent donc à chaque pipeline**.

```bash
# En local, avec la pile démarrée :
DATABASE_URL="postgresql://scoring:scoring_dev@127.0.0.1:5432/monitoring" \
  uv run pytest --cov=api
```
