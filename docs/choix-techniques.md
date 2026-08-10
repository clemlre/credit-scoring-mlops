# Choix techniques et alternatives écartées

L'énoncé laisse le choix des outils, à condition de pouvoir les justifier. Ce document
répond, pour chaque décision structurante, à trois questions : **qu'a-t-on choisi**,
**qu'a-t-on écarté**, et **à quelle condition ce choix deviendrait mauvais**.

---

## 1. FastAPI plutôt que Gradio ou Streamlit pour l'API

Les ressources du projet présentent Streamlit et Gradio ; l'énoncé mentionne
« Gradio ou FastAPI » pour l'API elle-même.

**Choisi : FastAPI.**

| Critère | FastAPI | Gradio | Streamlit |
|---|---|---|---|
| Vocation première | API HTTP | démo interactive de modèle | application de données |
| Contrat d'entrée typé | Pydantic, validation stricte | typage faible des composants | pas de notion de contrat |
| Documentation des routes | OpenAPI/Swagger générée | non | non |
| Codes de statut HTTP | natifs et maîtrisés | limités | sans objet |
| Consommation machine à machine | conçu pour | possible via `/api/predict` | inadapté |

Le besoin exprimé par Chloé est que le département « Crédit Express » traite des
demandes **en quasi temps réel** : le client de cette API est un système d'information,
pas un humain devant un navigateur. Or l'indicateur d'évaluation « j'ai documenté les
routes » et le point de vigilance « appréhender les erreurs correctement » supposent des
routes, des schémas et des codes de statut — c'est-à-dire exactement ce que FastAPI
fournit et ce que Gradio n'a pas vocation à offrir.

**Ce qui rendrait ce choix mauvais** : si le livrable attendu était une démonstration
visuelle destinée à un utilisateur métier plutôt qu'un service consommé par un SI. Dans
ce cas Gradio serait plus direct, au prix de la documentation des routes et de la
finesse des erreurs.

**Où Streamlit et Gradio gardent leur place** : à l'étape 3, pour le tableau de bord de
monitoring (distribution des scores, latence, dérive), qui est un usage visuel et humain.
Les ressources du projet les présentent d'ailleurs comme un moyen de « tester votre API »
et de « visualiser l'analyse de drift » — pas comme un substitut à l'API.

---

## 2. Un dictionnaire libre de features plutôt qu'un identifiant client

Le modèle consomme 779 features agrégées sur 7 tables, pas les données brutes d'un
demandeur. Trois contrats étaient possibles.

| Option | Pourquoi écartée |
|---|---|
| Exiger les 779 features | Payload de plusieurs centaines de champs obligatoires, inutilisable en pratique et impossible à remplir pour un primo-emprunteur. |
| `SK_ID_CURR` + recherche dans un magasin de features | Élégant côté appelant, mais impose d'embarquer des données clients dans l'image ou d'ajouter une base — donc des données personnelles dans un livrable public. |
| **Dictionnaire libre (retenu)** | L'appelant transmet ce dont il dispose ; les features absentes sont traitées comme manquantes, ce que LightGBM sait faire nativement. |

**Ce qui rendrait ce choix mauvais** : en production réelle avec un magasin de features
déjà en place, l'option `SK_ID_CURR` deviendrait préférable — l'API n'aurait plus à faire
confiance à l'appelant sur le calcul des agrégats.

---

## 3. Format texte natif LightGBM plutôt qu'un pickle

| Critère | Texte natif | Pickle |
|---|---|---|
| Exécution de code au chargement | non | **oui** — un pickle d'origine inconnue est un vecteur d'exécution |
| Sensibilité aux versions | tolérant entre versions mineures | exige les mêmes versions de scikit-learn/LightGBM |
| Dépendances au chargement | LightGBM seul | scikit-learn complet |
| Lisibilité | texte, inspectable | binaire opaque |

Conséquence mesurée : l'image de production n'a besoin ni de scikit-learn, ni de pandas,
ni de pyarrow — **951 Mo → 563 Mo**.

---

## 4. Bornes de validation mesurées, pas décrétées

Le point de vigilance de l'énoncé cite « un âge de -5 ans » comme exemple de valeur hors
plage. Appliqué naïvement ici, ce critère serait **faux** : dans ce jeu de données,
`DAYS_BIRTH` vaut −9461 pour un client de 26 ans, car les durées comptent les jours
*avant* la demande. Une règle « l'âge doit être positif » rejetterait 100 % des dossiers
valides.

Les bornes ont donc été mesurées sur les 307 507 clients d'entraînement avant d'être
codées, puis vérifiées : **0 faux rejet sur 50 000 clients réels**.

| Famille | Règle | Mesure ayant servi de base |
|---|---|---|
| `EXT_SOURCE_*` | ∈ [0, 1] | min 8,2e-08 — max 0,963 |
| `DAYS_*` (hors `_PERC`) | ≤ 0 | jours avant la demande |
| `AMT_*`, `CNT_*` | ≥ 0 | min observé 0,0 |
| `FLAG_*` | ∈ {0, 1} | aucune autre valeur |

Même démarche pour le plancher de complétude : il porte sur les **245 features de
dossier** et non sur les 779, parce que les 534 agrégats d'historique manquent
légitimement chez un primo-emprunteur. Seuil à 50 %, quand le 1er centile observé est à
78,4 % — la marge est délibérée.

---

## 5. Pertinence des tests

L'indicateur d'évaluation demande de « justifier de la pertinence des tests ». Le choix
n'a pas été de maximiser un pourcentage, mais de couvrir les trois cas critiques cités
par l'énoncé, puis les modes de défaillance propres à un service de scoring.

| Catégorie | Ce qui est testé | Pourquoi c'est critique |
|---|---|---|
| Champs manquants | dossier trop incomplet, payload vide, corps sans champ | un score calculé sur un dossier vide a l'apparence d'une prédiction |
| Valeurs hors plage | 5 familles, dans les deux sens | une erreur d'unité ou de signe passe sinon inaperçue |
| Types incorrects | texte, chaîne numérique, booléen, liste, objet | `"0,5"` lu selon une mauvaise locale fausse un revenu sans bruit |
| Valeurs non finies | `Infinity`, `-Infinity`, `NaN` | **a réellement fait tomber le service en 500** avant correction |
| Indisponibilité | modèle non chargé, erreur interne | doit dégrader proprement, sans fuite de trace |
| Fidélité au modèle | égalité stricte avec le booster source | une divergence silencieuse est le pire défaut possible |
| Cohérence métier | un score externe plus faible augmente le risque | détecte une inversion de signe ou un décalage de colonnes |

Le taux de couverture (100 %) est une **conséquence** de cette liste, pas son objectif.
Un plancher à 95 % est appliqué en CI pour interdire une dérive.

---

## 6. Démarrage en mode dégradé plutôt qu'échec immédiat

Si l'artefact est absent, le service démarre quand même et `/health` répond `503` en
nommant le fichier manquant.

**Alternative écartée** : échouer au démarrage (*fail fast*), qui est l'orthodoxie. Elle
a un défaut opérationnel : un conteneur en boucle de redémarrage n'expose aucun message
exploitable, alors qu'un service qui répond « voici ce qui me manque » se diagnostique en
une requête. Dans les deux cas la sonde de disponibilité retire l'instance du trafic, donc
aucun trafic n'est servi par erreur.

---

## 7. Artefact de modèle versionné dans Git

**Assumé comme un compromis, pas comme une bonne pratique.** Sans lui, ni la CI ni le
build Docker ne fonctionnent, puisque le registre MLflow de la Partie 1 vit hors dépôt.
Coût : 5,4 Mo, une version. En contexte industriel, l'artefact vivrait dans un registre
de modèles ou un magasin d'artefacts, et le pipeline l'y récupérerait au build.
