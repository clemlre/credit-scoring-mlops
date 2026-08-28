# Conception — Analyse du data drift

**Date :** 2026-08-28
**Livrable visé :** livrable 6, « Analyse du Data Drift **au format notebook** »
**Fichier produit :** `notebooks/07_data_drift.ipynb`

---

## 1. Objectif et exigences

Trois sources fixent le contenu attendu, et elles ne demandent pas la même chose.

**Le livrable** (`04 - Livrables.md`) impose le **format notebook**. Un tableau de bord
Streamlit ne s'y substitue pas ; il ne peut que s'y ajouter.

**La mission** (`03 - Mission`, étape 3) demande davantage que la dérive des features :

> un tableau de bord ou rapport de monitoring montrant des métriques clés (ex. :
> distribution des scores prédits, latence de l'API, temps d'inférence)

et

> la détection d'anomalies (taux d'erreur, latence anormale)

Elle attend en résultat « un script ou notebook réalisant l'analyse automatique des
**données stockées** » et « une présentation de l'étude sur la dérive des données et
les **points de vigilance** résultants ».

**La grille d'évaluation** retient deux indicateurs : *visualisation des différences
présente* et *différences interprétables*. Peu exigeante sur la forme, très exigeante
sur la capacité à lire les écarts.

L'analyse doit donc couvrir **la dérive des données, la dérive du score, et les
métriques opérationnelles**, et se terminer par des points de vigilance actionnables.

## 2. Contraintes retenues

| Contrainte | Origine | Conséquence |
|---|---|---|
| Notebook **autonome**, sans module externe | Décision projet | Tout le code d'analyse vit dans le `.ipynb`. Pas de package `monitoring/`. |
| Rédaction **sobre** | Décision projet | Cellules markdown courtes disant ce qu'on regarde et ce qu'on en conclut. Commentaires rares et utiles. Pas de ton didactique. |
| Aucune donnée client versionnée | `.gitignore`, licence Kaggle, dépôt public | Le notebook lit PostgreSQL en direct. Ses sorties sont sauvegardées dans le `.ipynb` pour que l'analyse soit lisible sans rejouer la pile. Aucun extrait de données n'est committé. |
| **Aucune destruction de données** | Décision projet | Le nouveau trafic s'ajoute à la suite de l'existant. Ni `down -v`, ni `TRUNCATE`. Le découpage se fait par fenêtres temporelles. |
| Image Docker inchangée | Étape 2 | Evidently entre dans un groupe de dépendances `monitoring`, jamais dans le cœur d'inférence. |

### Conséquence assumée

Le code d'analyse n'étant pas dans un module, **il n'est pas couvert par la CI**. Pour
compenser, le notebook intègre une **vérification de l'instrument de mesure** (§ 4.3) :
sans elle, rien ne prouverait que le calcul de dérive est juste.

## 3. Sources de données

### Référence — jeu d'entraînement de la Partie 1

`output/feature_dataset.parquet` du projet P6, hors dépôt (278 Mo). Lignes où
`TARGET` est renseigné : 307 507 dossiers.

**Échantillon déterministe de 20 000 lignes** (`random_state=42`). Raison : le PSI et
les tests de distribution se stabilisent bien avant 20 000 observations, alors que le
temps de calcul d'Evidently croît linéairement. L'échantillonnage est fixé pour que
deux exécutions donnent le même rapport.

### Courant — prédictions stockées en production

Table `predictions` de PostgreSQL. La colonne `features` (`JSONB`) contient le payload
tel que reçu ; les colonnes `probability`, `decision`, `latency_ms`,
`application_ratio` et `history_ratio` fournissent les métriques opérationnelles.

### Découpage en deux fenêtres

```
     lignes existantes           fenêtre A                fenêtre B
   (runs du 28/08 matin)      trafic normal            trafic décalé
   ─────────────────────┬────────────────────────┬────────────────────────▶
                        │                        │                     temps
                  DEBUT_ANALYSE              COUPURE
```

Trois constantes en tête de notebook : `DEBUT_ANALYSE`, `COUPURE`, `FIN_ANALYSE`.
Les lignes antérieures à `DEBUT_ANALYSE` sont exclues — elles proviennent de runs de
mise au point et mélangent les deux régimes.

Le notebook affiche un tableau des volumes par minute, de sorte que les frontières
soient **visuellement vérifiables** et non prises sur parole.

**Limite assumée.** Découper sur le temps est plus fragile qu'une colonne `tag` en
base. Ajouter cette colonne au schéma serait plus robuste, mais modifierait l'API pour
les besoins d'une analyse : le compromis penche du côté de ne pas toucher au service.
La procédure de génération ménage une pause de deux minutes entre les deux fenêtres,
ce qui rend la frontière non ambiguë.

## 4. Contenu du notebook

### 4.1 Sélection des features suivies

Le modèle compte 779 features. Les comparer toutes produirait un mur de graphiques
illisible, un temps de calcul inutile, et une centaine de faux positifs par simple
effet du nombre de tests.

**Deux niveaux :**

- **Vue d'ensemble** sur l'ensemble des features communes aux deux jeux : part de
  features détectées comme dérivées, sans détail individuel.
- **Analyse détaillée** sur les **20 features les plus importantes** du modèle,
  lues directement dans le booster LightGBM
  (`booster.feature_importance(importance_type="gain")`), et non recopiées depuis le
  notebook `06_feature_importance.ipynb` — la source de vérité est l'artefact servi.

**Justification à défendre :** une dérive sur une feature de poids nul ne déplace pas
le score. Trier par importance, c'est trier par impact métier.

### 4.2 Méthode de mesure

**PSI** (*Population Stability Index*), conformément à la documentation Evidently
fournie (`DataDriftPreset(method="psi")`) et à l'usage du scoring de crédit.

Seuils conventionnels retenus :

| PSI | Lecture |
|---|---|
| < 0,10 | population stable |
| 0,10 – 0,25 | dérive modérée, à surveiller |
| > 0,25 | dérive significative, action requise |

**Pourquoi pas le test de Kolmogorov-Smirnov** (défaut d'Evidently sur les variables
numériques) : sa p-value dépend de la taille d'échantillon. À 20 000 observations de
référence, un écart sans portée métier ressort « significatif ». Le PSI mesure une
amplitude, pas une significativité — c'est ce qu'on veut ici.

### 4.3 Vérification de l'instrument

Le code d'analyse n'étant pas testé en CI, le notebook valide sa propre mesure sur
deux cas de contrôle avant de conclure quoi que ce soit :

1. **Référence contre elle-même** (deux moitiés tirées au hasard) → PSI attendu
   proche de 0. Détecte une erreur d'alignement des colonnes ou de binning.
2. **Référence contre une version décalée d'une amplitude connue** → PSI attendu
   au-dessus du seuil de 0,25.

Si l'un des deux échoue, le reste du notebook n'a aucune valeur. Cette cellule est
donc placée **avant** les analyses, pas en annexe.

### 4.4 Plan des sections

| § | Contenu | Sortie attendue |
|---|---|---|
| 1 | Contexte, chargement des deux sources, volumes par fenêtre | Tableau des volumes par minute |
| 2 | Vérification de l'instrument (§ 4.3) | Deux PSI de contrôle |
| 3 | Sélection des features suivies (§ 4.1) | Top 20 par gain, avec leur poids |
| 4 | **Baseline** : production réelle vs entraînement | Rapport Evidently + part de features dérivées + distributions des features les plus décalées |
| 5 | **Scénario contrôlé** : fenêtre décalée, puis balayage d'amplitude | Courbe PSI en fonction du décalage, seuil de déclenchement |
| 6 | Dérive du **score prédit** et du taux de refus entre fenêtres | Distributions des probabilités, taux de refus comparés |
| 7 | **Métriques opérationnelles** : latence p50/p95/p99, couverture des dossiers, anomalies | Tableaux et graphiques |
| 8 | Conclusions et **points de vigilance** | Quoi surveiller, à quelle fréquence, seuils d'alerte, critère de réentraînement |

### 4.5 Dérive du taux de valeurs manquantes

Traitée comme une **dimension de dérive à part entière**, séparément des comparaisons
de distributions.

Motif : la production peut légitimement envoyer des dossiers sans historique de crédit
(primo-emprunteur) là où la référence en contient. Mélanger ce phénomène aux
distributions de valeurs produirait des dérives fantômes. Les colonnes
`application_ratio` et `history_ratio`, enregistrées à chaque prédiction, servent
exactement à ça.

### 4.6 Balayage d'amplitude

Réalisé **en mémoire**, sur les features de la fenêtre A, en appliquant un décalage
croissant (0,00 → 0,30 par pas de 0,05) aux scores externes.

Pourquoi pas cinq passages supplémentaires dans l'API : cela peuplerait la base de six
fenêtres à démêler, pour un résultat identique. Le décalage est une transformation
déterministe ; le faire transiter par HTTP n'apporte aucune information.

Le § 5 conserve **une** fenêtre réellement passée par l'API et stockée en base
(fenêtre B), qui prouve la chaîne complète API → PostgreSQL → analyse. Le balayage
complète, il ne remplace pas.

## 5. Génération du trafic

À exécuter avant le notebook, la pile étant démarrée :

```bash
# Fenêtre A — trafic normal
python scripts/simuler_trafic.py --nombre 3000 --lot 200

# Pause de 2 minutes : rend la frontière entre fenêtres non ambiguë

# Fenêtre B — trafic décalé
python scripts/simuler_trafic.py --nombre 1000 --lot 200 --decalage 0.2
```

> Ajouter `--url http://127.0.0.1:$API_PORT` si la pile n'écoute pas sur 8000
> (voir `docs/monitoring.md`). Sur le poste de développement actuel, `API_PORT=8001`.

Volume attendu : ~4 000 lignes supplémentaires, soit **≈ 38 Mo** (mesure de référence :
9,5 ko par prédiction, index compris).

Les horodatages de début, de coupure et de fin sont relevés à l'issue de ces commandes
et reportés dans les constantes du notebook.

## 6. Dépendances

Nouveau groupe `monitoring` dans `pyproject.toml` :

```toml
monitoring = [
    "evidently==0.7.21",
]
```

Résolution vérifiée : compatible avec `numpy<2.3` (résout en 2.2.6), `pandas==2.3.3`,
`scikit-learn==1.8.0`, `mlflow==3.12.0`, `shap==0.51.0`. 765 paquets résolus sans
conflit.

**Hors du cœur d'inférence** : l'image Docker reste à 583 Mo. Evidently n'est jamais
importé par `api/`.

## 7. Gestion des erreurs

| Situation | Comportement attendu |
|---|---|
| Parquet de la Partie 1 absent | Message explicite indiquant le chemin attendu et la variable `P6_PROJECT_ROOT`, puis arrêt propre |
| PostgreSQL injoignable | Message indiquant la commande de démarrage de la pile, puis arrêt propre |
| Fenêtre vide ou trop petite | Message indiquant les volumes trouvés et la commande de génération de trafic |
| Feature absente d'un des deux jeux | Exclue de la comparaison, et **comptabilisée** dans un tableau des exclusions — une feature manquante est une information, pas un détail |

Aucun de ces cas ne doit produire une trace Python. Un notebook livré doit rester
lisible même lorsqu'il ne peut pas s'exécuter.

## 8. Hors périmètre

- **Tableau de bord Streamlit.** Le livrable imposé est le notebook. Un dashboard
  pourra être ajouté ensuite ; le notebook n'y fait pas obstacle.
- **Alerting automatique.** Le notebook définit des seuils et les documente ; il ne
  branche pas de notification.
- **Réentraînement.** Le § 8 énonce le critère de déclenchement, il ne l'implémente pas.
- **Journalisation des requêtes refusées (422).** Absente du stockage par choix
  d'étape 3.1 ; le notebook le signale comme angle mort du dispositif.
- **Dérive de la performance du modèle** (*model drift*). Elle exigerait de connaître
  le défaut réel des clients scorés, information qui n'arrive que des mois plus tard.
  Le notebook explique pourquoi c'est hors d'atteinte ici, plutôt que de faire comme
  si le sujet n'existait pas.

### Sort du répertoire `monitoring/`

Le README l'annonce comme destination des « logs de production et rapports de drift ».
Le stockage étant finalement en base et l'analyse dans le notebook, **il reste vide
pour ce livrable**, et le README est corrigé en conséquence : sa description promet
aujourd'hui quelque chose qui n'arrivera pas.

Le rapport Evidently **n'est pas exporté en HTML dans le dépôt**. Il agrège des
distributions de dossiers Home Credit ; même agrégées, ces données n'ont pas leur place
dans un dépôt public sous licence Kaggle. Les résultats sont lisibles dans les sorties
sauvegardées du notebook. Un export HTML local reste possible pour la soutenance, à
condition qu'il ne soit pas versionné — `.gitignore` couvre déjà `monitoring/logs/`,
une entrée `monitoring/*.html` sera ajoutée par précaution.

## 9. Critères d'acceptation

Le livrable est terminé quand :

1. Le notebook s'exécute de bout en bout, sorties comprises, sur la pile locale.
2. Les deux cas de contrôle de l'instrument passent (§ 4.3).
3. Les huit sections du § 4.4 sont présentes et concluent chacune par une lecture,
   pas seulement un graphique.
4. Le § 8 énonce des points de vigilance actionnables : quoi surveiller, à quelle
   fréquence, à partir de quel seuil agir.
5. Aucune donnée client n'est versionnée.
6. `ruff check` passe sur le dépôt et la suite de tests existante reste verte.
7. Le README renvoie au notebook depuis la section monitoring, et sa description du
   répertoire `monitoring/` est corrigée (elle promet aujourd'hui des artefacts qui
   n'y seront pas).
