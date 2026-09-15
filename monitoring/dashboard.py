"""Tableau de bord de monitoring de l'API de scoring.

Trois questions, dans l'ordre de lecture : le modèle décide-t-il comme avant ? les
données ont-elles changé ? le service tient-il ? Un bandeau répond aux trois d'un
coup ; un onglet détaille chacune, plus un onglet sur le modèle en service.

Lit les tables `requests` et `predictions` alimentées par l'API, le profil de référence
de dérive (`profil_reference.json`) et la carte d'identité du modèle
(`models/model_metadata.json`).

    uv run --group monitoring streamlit run monitoring/dashboard.py

DATABASE_URL désigne la base (par défaut, celle de docker-compose.yml).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import psycopg
import streamlit as st

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from monitoring import indicateurs, queries  # noqa: E402

PROFIL = Path(__file__).with_name("profil_reference.json")
METADATA = RACINE / "models" / "model_metadata.json"

versions = st.cache_data(queries.versions, ttl=30, show_spinner=False)
resume_requetes = st.cache_data(queries.request_summary, ttl=30, show_spinner=False)
resume_predictions = st.cache_data(queries.prediction_summary, ttl=30, show_spinner=False)
serie_requetes = st.cache_data(queries.request_series, ttl=30, show_spinner=False)
serie_latence = st.cache_data(queries.latency_series, ttl=30, show_spinner=False)
serie_predictions = st.cache_data(queries.prediction_series, ttl=30, show_spinner=False)
erreurs = st.cache_data(queries.errors, ttl=30, show_spinner=False)
valeurs_production = st.cache_data(queries.production_values, ttl=30, show_spinner=False)

# Couleurs vérifiées pour les daltonismes et le contraste sur fond blanc : le bleu porte
# toute série neutre, l'orange une seconde série, l'ambre et le rouge ne disent que
# « attention » et « incident ». L'encre grise trace les repères (seuils, objectifs).
BLEU, ORANGE, AMBRE, ROUGE, ENCRE = "#2a78d6", "#eb6834", "#c98500", "#d03b3b", "#52514e"

_MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
         "septembre", "octobre", "novembre", "décembre"]
_JOURS = ["dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi"]
LOCALE_FR = {
    "number": {"decimal": ",", "thousands": " ", "grouping": [3], "currency": ["", " €"]},
    "time": {
        "dateTime": "%A %e %B %Y à %X",
        "date": "%d/%m/%Y",
        "time": "%H:%M:%S",
        "periods": ["", ""],
        "days": _JOURS,
        "shortDays": [j[:3] + "." for j in _JOURS],
        "months": _MOIS,
        "shortMonths": [m[:4] + "." if len(m) > 4 else m for m in _MOIS],
    },
}
AXE_TEMPS = alt.Axis(format="%d/%m %H:%M", labelAngle=0)

BOITES = {"ok": st.success, "info": st.info, "warning": st.warning, "error": st.error}
QUESTIONS = (
    "1. Le modèle décide-t-il comme avant ?",
    "2. Les données ont-elles changé ?",
    "3. Le service tient-il ?",
)


def format_ms(valeur) -> str:
    return "N/A" if valeur is None else f"{indicateurs.decimal(valeur, 1)} ms"


def format_pct(valeur) -> str:
    return "N/A" if valeur is None else indicateurs.pourcentage(valeur)


def format_decimal(valeur, chiffres: int = 3) -> str:
    return "N/A" if valeur is None else indicateurs.decimal(valeur, chiffres)


def carte(colonne, titre: str, valeur: str, aide: str, attendu: str | None = None) -> None:
    """Une valeur, son repère « attendu » juste dessous, et l'explication au survol."""
    colonne.metric(
        titre, valeur, delta=attendu, delta_color="off", delta_arrow="off", help=aide, border=True
    )


def definitions(*lignes: str) -> None:
    with st.expander("Définitions des termes de cet onglet"):
        st.markdown("\n".join(f"- {ligne}" for ligne in lignes))


def afficher(graphique) -> None:
    st.altair_chart(graphique.configure(locale=LOCALE_FR), width="stretch")


def repere(valeur: float, etiquette: str, y: float = 8) -> alt.LayerChart:
    """Ligne verticale de référence, avec son étiquette : un seuil, un objectif."""
    donnees = pd.DataFrame({"x": [valeur], "etiquette": [etiquette]})
    ligne = alt.Chart(donnees).mark_rule(color=ENCRE, strokeDash=[6, 4]).encode(x="x:Q")
    texte = (
        alt.Chart(donnees)
        .mark_text(align="left", dx=5, dy=-4, color=ENCRE, fontSize=11)
        .encode(x="x:Q", y=alt.value(y), text="etiquette:N")
    )
    return ligne + texte


def repere_horizontal(valeur: float, etiquette: str) -> alt.LayerChart:
    donnees = pd.DataFrame({"y": [valeur], "etiquette": [etiquette]})
    ligne = alt.Chart(donnees).mark_rule(color=ENCRE, strokeDash=[6, 4]).encode(y="y:Q")
    texte = (
        alt.Chart(donnees)
        .mark_text(align="left", dx=4, dy=-6, color=ENCRE, fontSize=11)
        .encode(x=alt.value(0), y="y:Q", text="etiquette:N")
    )
    return ligne + texte


def graphique_activite(serie: pd.DataFrame) -> alt.Chart:
    couleurs = alt.Scale(domain=["2xx", "4xx", "5xx"], range=[BLEU, AMBRE, ROUGE])
    return (
        alt.Chart(serie)
        .mark_bar()
        .encode(
            x=alt.X("instant:T", title=None, axis=AXE_TEMPS),
            y=alt.Y("sum(appels):Q", title="Appels"),
            color=alt.Color("classe:N", scale=couleurs, title="Réponse"),
            tooltip=["instant:T", "classe:N", "sum(appels):Q"],
        )
        .properties(height=240)
    )


def graphique_latence(serie: pd.DataFrame) -> alt.LayerChart:
    centiles = serie.melt(
        id_vars="instant", value_vars=["p50", "p95"], var_name="centile", value_name="ms"
    )
    lignes = (
        alt.Chart(centiles)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("instant:T", title=None, axis=AXE_TEMPS),
            y=alt.Y("ms:Q", title="Durée de POST /predict (ms)"),
            color=alt.Color(
                "centile:N", scale=alt.Scale(domain=["p50", "p95"], range=[BLEU, ORANGE]),
                title="Centile",
            ),
            tooltip=["instant:T", "centile:N", alt.Tooltip("ms:Q", format=".2f")],
        )
    )
    objectif = indicateurs.SEUIL_LATENCE_P95_MS
    if centiles["ms"].max() >= objectif * 0.6:
        lignes = lignes + repere_horizontal(objectif, f"objectif p95 : {objectif:.0f} ms")
    return lignes.properties(height=240)


def graphique_scores(probabilites: pd.Series, seuil: float) -> alt.LayerChart:
    donnees = pd.DataFrame({"probabilite": probabilites.clip(0, 1)})
    histogramme = (
        alt.Chart(donnees)
        .mark_bar(color=BLEU)
        .encode(
            x=alt.X("probabilite:Q", bin=alt.Bin(step=0.02), title="Probabilité de défaut"),
            y=alt.Y("count():Q", title="Dossiers"),
            tooltip=[alt.Tooltip("probabilite:Q", bin=alt.Bin(step=0.02)), "count():Q"],
        )
    )
    etiquette = f"seuil {indicateurs.decimal(seuil)} : refus à droite"
    return (histogramme + repere(seuil, etiquette)).properties(height=240)


def graphique_refus(serie: pd.DataFrame, attendu: float | None) -> alt.LayerChart:
    courbe = (
        alt.Chart(serie)
        .mark_line(point=True, color=BLEU, strokeWidth=2)
        .encode(
            x=alt.X("instant:T", title=None, axis=AXE_TEMPS),
            y=alt.Y("taux_refus:Q", title="Taux de refus", axis=alt.Axis(format=".0%")),
            tooltip=["instant:T", alt.Tooltip("taux_refus:Q", format=".1%"), "predictions:Q"],
        )
    )
    if attendu is not None:
        courbe = courbe + repere_horizontal(attendu, f"attendu : {format_pct(attendu)}")
    return courbe.properties(height=240)


def graphique_derive(tableau: pd.DataFrame) -> alt.LayerChart:
    couleurs = alt.Scale(
        domain=["stable", "modérée", "significative"], range=[BLEU, AMBRE, ROUGE]
    )
    barres = (
        alt.Chart(tableau)
        .mark_bar()
        .encode(
            x=alt.X("psi:Q", title="PSI (0 = même distribution qu'à l'entraînement)"),
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("bande:N", scale=couleurs, title="Dérive"),
            tooltip=["feature:N", alt.Tooltip("psi:Q", format=".3f"), "bande:N"],
        )
    )
    # Les deux bornes des bandes de la légende ; trop proches pour porter une étiquette
    # lisible quand une feature dérive fortement, la légende suffit.
    seuils = pd.DataFrame({"x": [indicateurs.PSI_STABLE, indicateurs.PSI_SIGNIFICATIF]})
    lignes = alt.Chart(seuils).mark_rule(color=ENCRE, strokeDash=[6, 4]).encode(x="x:Q")
    return (barres + lignes).properties(height=520)


def bandeau(etats: list[tuple[str, str]]) -> None:
    """Les trois réponses côte à côte : icône et texte, jamais la couleur seule."""
    for colonne, question, (niveau, texte) in zip(st.columns(3), QUESTIONS, etats, strict=True):
        with colonne:
            st.markdown(f"**{question}**")
            BOITES[niveau](texte)


def onglet_modele(metadata: dict, metriques: dict) -> None:
    seuil = float(metadata["decision_threshold"])
    st.markdown(
        "Le modèle servi et ses performances **mesurées hors échantillon** sur les "
        "307 507 clients de la Partie 1 (validation croisée, seuil de décision "
        f"{indicateurs.decimal(seuil)}). Ce sont les repères des autres onglets."
    )
    ligne = st.columns(4)
    carte(
        ligne[0], "Modèle servi", f"v{metadata['model_version']}",
        f"{metadata['n_trees']} arbres LightGBM, {metadata['n_features']} features.",
        attendu=f"seuil {indicateurs.decimal(seuil)}",
    )
    carte(
        ligne[1], "AUC", format_decimal(metriques["auc"]),
        "Capacité à classer un client qui fera défaut au-dessus d'un client sain. "
        "0,5 = hasard, 1 = parfait. Ne dépend pas du seuil.",
        attendu="hasard : 0,500",
    )
    carte(
        ligne[2], "Rappel des défauts", format_pct(metriques["rappel"]),
        "Part des clients qui feront défaut que le modèle refuse. C'est la métrique que le "
        "coût métier (10 × FN + 1 × FP) cherche à maximiser.",
        attendu=f"au seuil 0,5 : {format_pct(metriques['rappel_seuil_naif'])}",
    )
    carte(
        ligne[3], "F1", format_decimal(metriques["f1"]),
        "Moyenne harmonique de la précision et du rappel : un résumé en un chiffre.",
    )
    st.caption(
        f"Précision {format_pct(metriques['precision'])} (part des refus qui sont de vrais "
        f"défauts) · accuracy {format_pct(metriques['accuracy'])} · taux de refus attendu "
        f"{format_pct(metriques['taux_refus_attendu'])}. Le seuil {indicateurs.decimal(seuil)} "
        "refuse volontairement beaucoup de dossiers : un défaut accepté coûte dix fois un "
        f"bon client refusé. Au seuil 0,5 l'accuracy monterait à "
        f"{format_pct(metriques['accuracy_seuil_naif'])} mais le rappel tomberait à "
        f"{format_pct(metriques['rappel_seuil_naif'])} : le modèle n'attraperait presque "
        "aucun défaut. Détail : `docs/perf/metriques-modele.json`."
    )
    definitions(
        "**Hors échantillon (out-of-fold)** : chaque client est scoré par un modèle qui ne l'a "
        "pas vu à l'entraînement. Les chiffres reflètent ce qui se passe sur de nouveaux dossiers.",
        "**Seuil de décision** : la probabilité à partir de laquelle un dossier est refusé. "
        "Choisi pour minimiser le coût métier, pas pour maximiser l'accuracy.",
        "**Rappel** : défauts détectés / défauts réels. **Précision** : vrais défauts / "
        "dossiers refusés. **Accuracy** : décisions correctes / décisions.",
        "**Pourquoi l'accuracy trompe ici** : 8 % des clients font défaut ; accepter tout le "
        "monde donnerait déjà 92 % d'accuracy.",
    )


def onglet_decisions(predictions: dict, production, serie_pred, seuil: float, attendu) -> None:
    st.markdown(
        "**Ce que le métier voit en premier.** Un taux de refus qui s'écarte durablement du "
        "taux attendu signale un changement, du côté des dossiers ou du modèle : l'onglet "
        "suivant en cherche la cause."
    )
    ligne = st.columns(3)
    carte(ligne[0], "Dossiers scorés", indicateurs.entier(predictions["nombre"]),
          "Prédictions sur la période, lots compris.")
    carte(
        ligne[1], "Taux de refus", format_pct(predictions["taux_refus"]),
        f"Part des dossiers dont la probabilité de défaut dépasse {indicateurs.decimal(seuil)}.",
        attendu=None if attendu is None else f"attendu {format_pct(attendu)} ± 20 %",
    )
    carte(
        ligne[2], "Couverture des dossiers", format_pct(predictions["couverture_dossier"]),
        "Part des features de demande renseignées, en moyenne. Des dossiers plus incomplets "
        "changent les décisions sans que le modèle soit en cause.",
        attendu=f"historique {format_pct(predictions['couverture_historique'])}",
    )
    if production.empty:
        st.info("Aucune prédiction sur la période.")
    else:
        gauche, droite = st.columns(2)
        with gauche:
            st.markdown("**Distribution des scores**")
            afficher(graphique_scores(production["probability"], seuil))
        with droite:
            st.markdown("**Taux de refus dans le temps**")
            afficher(graphique_refus(serie_pred, attendu))
    definitions(
        "**Probabilité de défaut** : la sortie du modèle, entre 0 et 1.",
        "**Taux de refus** : part des dossiers au-dessus du seuil. Le taux attendu est celui "
        "mesuré hors échantillon sur les clients de la Partie 1.",
        "**Couverture** : part des features renseignées dans le dossier envoyé. Les features "
        "d'historique de crédit sont légitimement absentes pour un primo-emprunteur.",
    )


def onglet_donnees(derives: list[dict], profil: dict, nombre: int) -> None:
    st.markdown(
        "**La cause possible.** Le PSI compare la distribution de chaque feature reçue en "
        "production à celle du jeu d'entraînement : si les dossiers ne ressemblent plus à ceux "
        "sur lesquels le modèle a appris, ses décisions ne sont plus garanties."
    )
    significatives = sum(d["bande"] == "significative" for d in derives)
    moderees = sum(d["bande"] == "modérée" for d in derives)
    lisible = nombre >= indicateurs.MIN_PREDICTIONS_PSI
    ligne = st.columns(3)
    carte(
        ligne[0], "Dérive significative", str(significatives) if lisible else "N/A",
        f"Features dont le PSI dépasse {indicateurs.decimal(indicateurs.PSI_SIGNIFICATIF)}.",
        attendu="attendu 0",
    )
    carte(
        ligne[1], "Dérive modérée", str(moderees) if lisible else "N/A",
        f"Features dont le PSI est entre {indicateurs.decimal(indicateurs.PSI_STABLE)} et "
        f"{indicateurs.decimal(indicateurs.PSI_SIGNIFICATIF)} : à surveiller.",
        attendu=f"sur {len(derives)} features suivies",
    )
    carte(
        ligne[2], "Dossiers analysés", indicateurs.entier(nombre),
        "Le PSI par déciles n'est lisible qu'à partir de quelques centaines de dossiers.",
        attendu=f"minimum {indicateurs.MIN_PREDICTIONS_PSI}",
    )
    if not lisible:
        st.info(
            f"{indicateurs.entier(nombre)} prédictions sur la période : le PSI n'est calculé "
            f"qu'à partir de {indicateurs.MIN_PREDICTIONS_PSI}. Élargir la fenêtre."
        )
    else:
        tableau = pd.DataFrame(derives)
        afficher(graphique_derive(tableau))
        st.caption(
            f"Référence : {indicateurs.entier(profil['lignes'])} dossiers du jeu d'entraînement. "
            f"Les {len(derives)} features suivies portent "
            f"{indicateurs.pourcentage(profil['part_du_gain'])} du gain du modèle. Lignes "
            f"pointillées : {indicateurs.decimal(indicateurs.PSI_STABLE)} et "
            f"{indicateurs.decimal(indicateurs.PSI_SIGNIFICATIF)}, les bornes des bandes de la "
            "légende."
        )
        with st.expander("Détail par feature (PSI, manquants, poids dans le modèle)"):
            st.dataframe(
                tableau,
                hide_index=True,
                width="stretch",
                column_config={
                    "feature": "Feature",
                    "psi": st.column_config.NumberColumn("PSI", format="localized"),
                    "bande": "Dérive",
                    "manquant_reference": st.column_config.NumberColumn(
                        "Manquants (référence)", format="percent"
                    ),
                    "manquant_production": st.column_config.NumberColumn(
                        "Manquants (production)", format="percent"
                    ),
                    "importance": st.column_config.NumberColumn(
                        "Gain dans le modèle", format="localized"
                    ),
                },
            )
    definitions(
        "**PSI (Population Stability Index)** : écart entre deux distributions, découpées en "
        "déciles de la référence. 0 = identiques ; < 0,10 stable ; 0,10 à 0,25 modérée ; "
        "≥ 0,25 significative (bandes usuelles en scoring de crédit).",
        "**Référence** : les déciles de 20 000 dossiers d'entraînement, versionnés sans "
        "aucune ligne client (`monitoring/profil_reference.json`).",
        "**Features suivies** : les 20 qui pèsent le plus dans le modèle (gain LightGBM).",
        "**Manquants** : un taux de valeurs absentes qui change est une dérive à part entière.",
    )


def onglet_service(requetes: dict, predictions: dict, serie_req, serie_lat, periode) -> None:
    st.markdown(
        "**La santé du service lui-même.** Une hausse des 422 signale un appelant qui a changé "
        "son format ; toute 500 est un incident ; la latence est mesurée côté serveur, hors "
        "réseau."
    )
    appels = requetes["appels"]
    taux_erreur = (requetes["erreurs_4xx"] + requetes["erreurs_5xx"]) / appels if appels else None
    ligne = st.columns(3)
    carte(ligne[0], "Appels", indicateurs.entier(appels),
          "Appels des routes suivies (/predict, /predict/batch, /model/info, /features).",
          attendu=f"{indicateurs.entier(requetes['erreurs_5xx'])} erreur(s) interne(s)")
    carte(
        ligne[1], "Taux d'erreur", format_pct(taux_erreur),
        "Réponses 4xx (entrée refusée) et 5xx (erreur interne).",
        attendu=f"seuil d'alerte {format_pct(indicateurs.SEUIL_TAUX_4XX)}",
    )
    carte(
        ligne[2], "Latence p95 de /predict", format_ms(requetes["latence_p95_ms"]),
        "95 % des appels /predict sont traités en moins que cette durée, réseau exclu.",
        attendu=f"objectif {indicateurs.SEUIL_LATENCE_P95_MS:.0f} ms · "
        f"inférence seule {format_ms(predictions['inference_p50_ms'])}",
    )
    if serie_req.empty:
        st.info("Aucun appel sur la période.")
        return
    gauche, droite = st.columns(2)
    with gauche:
        st.markdown("**Appels et réponses**")
        afficher(graphique_activite(serie_req))
    with droite:
        st.markdown("**Latence de POST /predict**")
        if serie_lat.empty:
            st.info("Aucun appel réussi à /predict sur la période.")
        else:
            afficher(graphique_latence(serie_lat))
    detail = erreurs(periode)
    with st.expander("Erreurs par route"):
        if detail.empty:
            st.write("Aucune erreur sur la période.")
        else:
            st.dataframe(detail, hide_index=True, width="stretch")
    definitions(
        "**2xx / 4xx / 5xx** : réponse réussie / entrée refusée par la validation (422 : "
        "champ inconnu, valeur hors plage, dossier trop incomplet) / erreur interne.",
        "**p50, p95** : la médiane et le 95ᵉ centile des durées. Le p95 dit ce que vivent "
        "les appels les plus lents, la moyenne le cacherait.",
        "**Latence serveur** : de la réception de la requête à l'envoi de la réponse. "
        "**Inférence** : le seul appel au modèle, une petite part de ce temps.",
    )


def choisir_periode(maintenant: datetime) -> tuple[tuple, timedelta | None]:
    """Période précise lue dans l'URL (?du=&au=), sinon fenêtre glissante."""
    if "du" not in st.query_params:
        nom_fenetre = st.selectbox("Fenêtre", list(indicateurs.FENETRES), index=1)
        duree = indicateurs.FENETRES[nom_fenetre]
        st.caption("Période précise : ajouter ?du=2026-09-10T15:50Z&au=... à l'URL.")
        return (maintenant - duree if duree is not None else None, None), duree

    try:
        periode = indicateurs.periode_precise(
            st.query_params["du"], st.query_params.get("au"), maintenant
        )
    except ValueError as exc:
        st.error(f"Période invalide dans l'URL : {exc}")
        st.stop()
    debut, fin = (instant.astimezone() for instant in periode)
    st.write(f"Du {debut:%d/%m/%Y %H:%M} au {fin:%d/%m/%Y %H:%M} (heure locale)")
    if st.button("Revenir aux fenêtres glissantes", width="stretch"):
        st.query_params.clear()
        st.rerun()
    return periode, periode[1] - periode[0]


def main() -> None:
    st.set_page_config(page_title="Monitoring : scoring de crédit", layout="wide")
    st.title("Suivi du modèle de scoring en production")
    st.caption(
        "À lire dans l'ordre : les décisions (l'effet visible), puis les données (la cause "
        "possible), puis le service (la santé technique)."
    )

    try:
        liste_versions = versions()
    except psycopg.OperationalError as exc:
        st.error(f"Base de monitoring injoignable : {exc}")
        st.code("docker compose up -d db", language="bash")
        st.stop()

    # Arrondi à la minute : la clé de cache reste stable entre deux rafraîchissements.
    maintenant = datetime.now(UTC).replace(second=0, microsecond=0)

    with st.sidebar:
        st.header("Période")
        periode, duree = choisir_periode(maintenant)
        choix_version = st.selectbox("Version du modèle", ["Toutes", *liste_versions])
        if st.button("Rafraîchir", width="stretch"):
            st.cache_data.clear()
        st.caption(f"Base : {queries.DSN.rsplit('@', 1)[-1]}")

    version = None if choix_version == "Toutes" else choix_version
    pas = indicateurs.pas_temporel(duree)

    profil = json.loads(PROFIL.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    metriques = indicateurs.metriques_modele(metadata)
    seuil = float(metadata["decision_threshold"])
    noms = tuple(profil["features"])

    requetes = resume_requetes(periode)
    predictions = resume_predictions(periode, version)
    production = valeurs_production(periode, version, noms)
    derives = indicateurs.derive(profil, {n: production[n].to_numpy() for n in noms})
    serie_req = serie_requetes(periode, pas)
    serie_pred = serie_predictions(periode, version, pas)
    serie_lat = serie_latence(periode, pas)

    bandeau(
        [
            indicateurs.etat_decisions(predictions, metriques["taux_refus_attendu"]),
            indicateurs.etat_donnees(derives, predictions["nombre"]),
            indicateurs.etat_service(requetes),
        ]
    )

    modele, decisions, donnees, service = st.tabs(
        ["Le modèle en service", *QUESTIONS]
    )
    with modele:
        onglet_modele(metadata, metriques)
    with decisions:
        onglet_decisions(
            predictions, production, serie_pred, seuil, metriques["taux_refus_attendu"]
        )
    with donnees:
        onglet_donnees(derives, profil, predictions["nombre"])
    with service:
        onglet_service(requetes, predictions, serie_req, serie_lat, periode)

    with st.expander("Que faire de ce tableau de bord ?"):
        st.markdown(
            "- Taux de refus hors bande : vérifier d'abord la couverture des dossiers "
            "(onglet 1), puis la dérive (onglet 2). Des dossiers plus incomplets changent les "
            "décisions sans que le modèle soit en cause.\n"
            "- Dérive significative : analyser la période dans "
            "`notebooks/07_data_drift.ipynb` (lien `?du=...&au=...` de la barre latérale).\n"
            "- Erreurs : une hausse des 422 se règle avec l'appelant ; toute 500 est un "
            "incident à investiguer par `request_id` (en-tête `X-Request-ID`).\n"
            "- Réentraîner se décide sur une baisse de performance mesurée quand les "
            "défauts réels sont connus, pas sur un PSI seul."
        )


if __name__ == "__main__":
    main()
