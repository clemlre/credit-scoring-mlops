# Analyse et optimisation des performances

Ce rapport couvre l'étape 4 : mesurer où part le temps d'une requête, corriger les goulots
trouvés, et vérifier que les corrections n'ont rien cassé. Tous les chiffres cités viennent
des fichiers de `docs/perf/`, produits par les scripts listés en fin de document.

## Résultat

Image d'origine contre image optimisée, dans les conditions du Space Hugging Face
« cpu-basic » : conteneur Linux limité à 2 CPU. Médiane de 3 tours alternés.

**Sous charge** — test oha sur `/predict`, 15 s par mesure (`docs/perf/charge.json`) :

| | Avant | Après | Écart |
|---|---:|---:|---:|
| Débit, 8 connexions | 187 req/s | 417 req/s | ×2,2 |
| Latence p95, 8 connexions | 84,0 ms | 41,0 ms | −51 % |
| Débit, 32 connexions | 219 req/s | 387 req/s | ×1,8 |
| Latence p95, 32 connexions | 309,8 ms | 136,6 ms | −56 % |
| Latence p99, 32 connexions | 362,6 ms | 164,9 ms | −55 % |

**Requête seule** — 1 000 requêtes séquentielles (`docs/perf/comparaison-finale.json`) :

| | Avant | Après | Écart |
|---|---:|---:|---:|
| Temps serveur, p50 | 3,07 ms | 2,19 ms | −29 % |
| Temps serveur, p95 | 7,57 ms | 4,33 ms | −43 % |
| Latence client, p50 | 6,56 ms | 5,31 ms | −19 % |

Le temps serveur est la durée que l'API journalise elle-même (champ `duration_ms` de la
table `requests`) ; il ne dépend pas du réseau. La latence client ajoute le réseau de Docker
Desktop, environ 3 ms sur ce poste.

**Lot de 200 dossiers** (`docs/perf/charge-lot200.json`) : 96 ms avant, 81 ms après.

**Pas de régression** : la suite de tests complète passe, dont la fidélité exacte au modèle sur des
clients réels. Les probabilités renvoyées par les deux images sont identiques au bit près
sur 20 dossiers comparés : le modèle et son format n'ont pas changé.

## Méthode

- **Conditions** : l'image de production, lancée avec `--cpus=2 --memory=4g`, sans base de
  données (comme sur le Space). Dossier envoyé : l'exemple « refusé » de Swagger, 245
  features.
- **Alternance** : les conteneurs sont mesurés tour à tour, trois fois, et on retient la
  médiane. Sur un portable, deux mesures successives du même conteneur varient facilement de
  20 % : mesurer « avant » un jour et « après » le lendemain ne prouverait rien.
- **Charge** : oha tourne dans un conteneur qui partage l'espace réseau de la cible. Un
  premier générateur en Python, sous Windows, donnait des débits trop instables (de 218 à
  367 req/s pour le même conteneur) pour attribuer un gain à un changement précis.
- **Instrument** : le temps serveur est lu dans les journaux JSON du conteneur. C'est le
  dispositif de monitoring de l'étape 3 qui sert ici de mesure.

Les mesures d'étapes en processus (`scripts/benchmark.py etapes`) sont faites sous Windows,
avec 14 cœurs : elles comparent des versions du code entre elles, elles ne prédisent pas la
latence en production.

## Point de départ

### Ce que disaient les mesures

Coût de chaque étape de `/predict`, en processus (`docs/perf/avant-etapes.json`) :

| Étape | p50 |
|---|---:|
| Parsing et validation Pydantic | 0,10 ms |
| Contrôle du contrat (noms, plages, couverture) | 0,34 ms |
| Inférence (`model.predict`) | 0,52 ms |
| Route complète, hors HTTP | 1,60 ms |

En HTTP, la même requête prenait 9,9 ms en médiane (`docs/perf/avant-http.json`). Le modèle
ne représentait qu'une petite part du temps de réponse : commencer par optimiser
l'inférence aurait été optimiser au mauvais endroit.

### Ce que disait cProfile

Sur la logique de la route (`scripts/benchmark.py profil`), deux fonctions dominaient :
l'appel LightGBM et `out_of_range_features`, avec **2 352 000 appels à `str.startswith` pour
2 000 requêtes**. La couverture du dossier était aussi calculée deux fois par requête.

cProfile ne voit que le thread dans lequel il tourne. La route étant synchrone, FastAPI
l'exécute dans un pool de threads : pour voir la pile HTTP, il a fallu profiler à part la
boucle d'événements d'uvicorn. On y voyait le coût de `BaseHTTPMiddleware` et de nombreux
passages par `run_in_threadpool` à chaque requête : la dépendance `get_model`, la route, la
sérialisation de la réponse et deux tâches de journalisation.

## Goulot 1 — le contrôle des plages de valeurs

Chaque valeur reçue était comparée aux cinq préfixes des règles (`EXT_SOURCE_`, `DAYS_`…).
Les bornes de chaque feature sont désormais résolues une fois, au chargement du modèle ; le
contrôle devient une recherche dans un dictionnaire. La couverture calculée pendant la
validation est transmise à `predict` au lieu d'être recalculée.

| En processus (`avant-etapes.json` → `apres-etapes.json`) | Avant | Après |
|---|---:|---:|
| Contrôle du contrat, p50 | 0,34 ms | 0,06 ms |
| Route hors HTTP, p50 | 1,60 ms | 0,62 ms |

Le nombre d'appels de fonctions mesuré par cProfile sur 2 000 requêtes passe de 3,66 à 1,68
million.

## Goulot 2 — la pile HTTP

- Le middleware `@app.middleware("http")` est un `BaseHTTPMiddleware`, qui crée un groupe de
  tâches et des flux mémoire à chaque requête. Il est remplacé par un middleware ASGI pur,
  `api/tracking.py`.
- La journalisation passait par deux tâches d'arrière-plan (les prédictions, puis la
  requête) : deux passages par le pool de threads et deux connexions à la base. Le
  middleware écrit maintenant la requête et ses prédictions en une seule transaction,
  après l'envoi de la réponse.
- La dépendance `get_model`, qui ne bloque jamais, devient `async` : un saut de thread de
  moins.

Effet, code seul et un worker (`docs/perf/comparaison-code.json`) : temps serveur p50
2,51 → 2,10 ms, p99 9,5 → 5,6 ms.

## Goulot 3 — les threads OpenMP dans un conteneur

Ce goulot ne se voyait pas dans les profils Python.

Dans un conteneur limité à 2 CPU, Python voit quand même tous les cœurs de l'hôte
(`os.cpu_count()` renvoie 14). LightGBM ouvre donc 14 threads OpenMP pour scorer **un seul**
dossier. Ces threads attendent activement, épuisent le quota de 2 CPU et le noyau bride le
conteneur jusqu'à la période suivante de l'ordonnanceur (100 ms).

Appel `booster.predict` sur un dossier, dans l'image, avec `--cpus=2`
(`docs/perf/openmp-conteneur.txt`) :

| `OMP_NUM_THREADS` | p50 |
|---|---:|
| 14 (défaut) | 102 ms |
| 2 | 0,08 à 0,14 ms |
| 1 | 0,06 à 0,20 ms |

Dans l'API, l'effet est moins brutal car les requêtes n'arrivent pas en rafale continue,
mais il reste net. Image optimisée avec 14 threads, puis avec 1 thread :

| | 14 threads | 1 thread | Source |
|---|---:|---:|---|
| Temps serveur p95, requête seule | 6,45 ms | 3,05 ms | `comparaison-openmp.json` |
| Débit, 8 connexions | 337 req/s | 417 req/s | `charge.json` |
| Latence p95, 32 connexions | 191 ms | 137 ms | `charge.json` |

L'image fixe `OMP_NUM_THREADS=1`. Contrepartie : un lot de 200 dossiers ne profite plus du
parallélisme dans LightGBM ; il prend 81 ms au lieu de 71 ms avec 14 threads
(`charge-lot200.json`), mais reste plus rapide que dans l'image d'origine (96 ms). L'usage
principal du service est le scoring unitaire « quasi temps réel » de Crédit Express : le
compromis est assumé.

## Exploiter les 2 vCPU : deux workers

Un seul processus uvicorn n'exécute du code Python que sur un cœur à la fois (GIL). Le
Space en a deux : l'image lance deux workers (`WEB_CONCURRENCY=2`, variable lue nativement
par uvicorn). Chaque worker charge son propre modèle ; la mémoire du conteneur passe
d'environ 75 à 180 Mo, négligeable face aux 16 Go du Space.

Avec deux workers et un thread OpenMP chacun, le service utilise exactement ses 2 vCPU,
sans sursouscription. L'effet des workers seuls n'est pas isolé proprement : la seule série
qui le mesure (`comparaison-workers.json`, 320 → 364 req/s) a été faite avec le générateur
Python, trop instable. Le gain combiné du code et des workers se lit dans `charge.json` :
187 → 337 req/s à 8 connexions, avant même le réglage d'OpenMP.

## ONNX Runtime : testé, non retenu

`scripts/evaluer_onnx.py` convertit le modèle avec onnxmltools, puis compare ONNX Runtime à
LightGBM, à un thread : sous Windows avec les données réelles (`docs/perf/onnx.json`), et
dans un conteneur Linux limité à 2 CPU (`docs/perf/onnx-linux.json`, option
`--sans-donnees`).

**Vitesse** (appel au modèle seul, p50) :

| | LightGBM | ONNX Runtime |
|---|---:|---:|
| 1 dossier, conteneur Linux 2 CPU | 0,088 ms | 0,030 ms |
| Lot de 200 dossiers, conteneur Linux 2 CPU | 9,0 ms | 5,7 ms |
| 1 dossier, Windows | 0,29 ms | 0,15 ms |
| Lot de 200 dossiers, Windows | 37,4 ms | 18,3 ms |

**Fidélité**, sur 100 000 dossiers réels de la Partie 1 :

| | |
|---|---:|
| Écart maximal de probabilité | 7,1 × 10⁻³ |
| Écart moyen | 1,7 × 10⁻⁶ |
| Dossiers avec un écart supérieur à 10⁻⁴ | 183 |
| Décisions changées au seuil de 0,10 | 0 |

L'écart vient du convertisseur, qui n'accepte que des entrées float32 : les seuils des arbres
sont arrondis, et une valeur très proche d'un seuil peut basculer dans l'autre branche.

**Pourquoi il n'est pas déployé :**

1. Le gain ne porte pas là où est le temps. Sur une requête unitaire, l'appel au modèle
   coûte 0,09 ms sur un temps serveur de 2,2 ms : ONNX ferait gagner 0,06 ms, environ 3 %.
   Même pour un lot de 200 dossiers, le modèle ne pèse que 9 ms sur les 81 ms de la
   requête ; le reste est le parsing et la validation de 49 000 valeurs.
2. Le service rendrait des probabilités qui ne sont plus celles du modèle validé à la
   Partie 1. Aucune décision ne change sur 100 000 dossiers, mais 183 scores bougent de plus
   de 10⁻⁴. Pour une décision de crédit, la reproductibilité du score servi compte plus
   qu'un gain que personne ne percevra.
3. Une étape de conversion et une dépendance de plus à maintenir.

**Quand y revenir** : si le modèle devenait nettement plus lourd (plus d'arbres, plus
profonds), au point que l'inférence redevienne une part significative de la requête. Il
faudrait alors faire valider l'écart de score par le métier avant la mise en production.

## Quantification

La quantification d'ONNX Runtime (int8 dynamique ou statique) réduit la précision des poids
des opérateurs `MatMul`, `Conv`, `Gemm`. Le graphe ONNX du modèle ne contient que `Cast`,
`Identity`, `Mul` et `TreeEnsembleClassifier` : il n'a aucun poids à quantifier. La seule
réduction de précision possible sur un ensemble d'arbres est le passage en float32 des
seuils, c'est-à-dire ce que fait déjà la conversion ONNX, dont l'effet est mesuré ci-dessus.

## Et en production ?

Même mesure depuis le poste de développement contre le Space, avant et après le
déploiement de cette version (400 requêtes séquentielles, puis 400 avec 8 clients) :

| | Avant (`hf-avant.json`) | Après (`hf-apres.json`) |
|---|---:|---:|
| p50, séquentiel | 110,4 ms | 111,7 ms |
| p95, séquentiel | 148,7 ms | 121,9 ms |
| Débit, 8 clients | 51,4 req/s | 50,6 req/s |

Aucune différence visible, et c'est attendu : le temps serveur ne représente que 2 à 3 ms
de ces 110 ms, le reste est le réseau entre le poste et Hugging Face. Avec 8 clients, le
débit est borné par ce même aller-retour réseau (8 connexions × ~9 réponses par seconde),
pas par le service. L'optimisation ne se voit donc qu'au plus près du serveur, ce que
mesurent les tests en conteneur ci-dessus : elle compte sous charge et pour le coût (à
trafic égal, deux fois moins de CPU), pas pour un utilisateur isolé.

Pour la relancer :

```bash
uv run python scripts/benchmark.py http https://clemlre-credit-scoring-api.hf.space \
  --repetitions 400 --concurrence 8 --sortie docs/perf/hf-apres.json
```

## Configuration finale

| Élément | Choix | Raison |
|---|---|---|
| Serveur | uvicorn, 2 workers | un processus par vCPU du Space |
| Inférence | LightGBM natif, `OMP_NUM_THREADS=1` | un thread par worker respecte le quota CPU ; ONNX ne gagnerait qu'environ 3 %, au prix de scores approchés |
| Middleware | ASGI pur | pas de groupe de tâches ni de flux mémoire par requête |
| Journalisation | une transaction par requête, après la réponse | un seul passage par le pool de threads |
| Matériel | CPU, 2 vCPU (« cpu-basic ») | environ 400 req/s mesurés ; un GPU n'apporte rien à un ensemble d'arbres de cette taille |

**Montée en charge** : l'API est sans état. Au-delà de ce qu'un conteneur absorbe, on ajoute
des réplicas derrière un répartiteur. La limite suivante serait PostgreSQL : chaque worker
ouvre un pool de 4 connexions au plus, soit 8 par réplica.

## Reproduire

```bash
# Étapes et profil, en processus
uv run python scripts/benchmark.py etapes --repetitions 2000
uv run python scripts/benchmark.py profil --repetitions 2000 --dump route.prof

# Conteneurs limités à 2 CPU
docker build -t credit-scoring-api:final .
docker run -d --name bench-final --cpus=2 --memory=4g -p 8014:8000 credit-scoring-api:final
docker run -d --name bench-avant --cpus=2 --memory=4g -p 8010:8000 <image de référence>

# Requête seule : latence client et temps serveur lu dans les journaux
uv run python scripts/comparer_images.py avant=bench-avant:8010 apres=bench-final:8014 \
  --tours 3 --sortie docs/perf/comparaison-finale.json

# Charge (oha dans le réseau du conteneur)
uv run python scripts/tester_charge.py bench-avant bench-final --connexions 8 32 \
  --sortie docs/perf/charge.json

# ONNX (fidélité : nécessite les données de la Partie 1)
uv sync --group perf
uv run --group perf python scripts/evaluer_onnx.py --lignes 100000 --sortie docs/perf/onnx.json
```

Un fichier `.prof` s'ouvre avec `python -m pstats route.prof`, ou avec snakeviz.
