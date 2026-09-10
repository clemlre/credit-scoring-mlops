"""Tableau de bord de monitoring de l'API de scoring.

Lit les tables `requests` et `predictions` alimentées par l'API.

    uv run --group monitoring streamlit run monitoring/dashboard.py

DATABASE_URL désigne la base (par défaut, celle de docker-compose.yml).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import psycopg
import streamlit as st
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring import indicateurs

DSN = os.environ.get("DATABASE_URL", "postgresql://scoring:scoring_dev@127.0.0.1:5432/monitoring")
PROFIL = Path(__file__).with_name("profil_reference.json")
SEUIL_DECISION = 0.10

_MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
         "septembre", "octobre", "novembre", "décembre"]
_JOURS = ["dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi"]
LOCALE_FR = {
    "number": {"decimal": ",", "thousands": "\u202f", "grouping": [3], "currency": ["", " €"]},
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


def lire(requete, parametres=()) -> pd.DataFrame:
    with psycopg.connect(DSN, connect_timeout=5) as connexion:
        curseur = connexion.execute(requete, parametres)
        colonnes = [c.name for c in curseur.description]
        return pd.DataFrame(curseur.fetchall(), columns=colonnes)


def premiere_ligne(requete, parametres) -> dict:
    ligne = lire(requete, parametres).iloc[0]
    return {k: (None if pd.isna(v) else v) for k, v in ligne.items()}


def filtre(periode: tuple, version: str | None = None) -> tuple:
    depuis, jusqua = periode
    conditions, parametres = [sql.SQL("TRUE")], []
    if depuis is not None:
        conditions.append(sql.SQL("occurred_at >= %s"))
        parametres.append(depuis)
    if jusqua is not None:
        conditions.append(sql.SQL("occurred_at < %s"))
        parametres.append(jusqua)
    if version is not None:
        conditions.append(sql.SQL("model_version = %s"))
        parametres.append(version)
    return sql.SQL(" AND ").join(conditions), parametres


@st.cache_data(ttl=30, show_spinner=False)
def versions() -> list[str]:
    resultat = lire("SELECT DISTINCT model_version FROM predictions ORDER BY 1")
    return resultat["model_version"].tolist()


@st.cache_data(ttl=30, show_spinner=False)
def resume_requetes(periode: tuple) -> dict:
    where, parametres = filtre(periode)
    return premiere_ligne(
        sql.SQL("""
            SELECT count(*) AS appels,
                   count(*) FILTER (WHERE status_code BETWEEN 400 AND 499) AS erreurs_4xx,
                   count(*) FILTER (WHERE status_code >= 500) AS erreurs_5xx,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms)
                       FILTER (WHERE path = '/predict') AS latence_p50_ms,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)
                       FILTER (WHERE path = '/predict') AS latence_p95_ms
            FROM requests WHERE {where}
        """).format(where=where),
        parametres,
    )


@st.cache_data(ttl=30, show_spinner=False)
def resume_predictions(periode: tuple, version: str | None) -> dict:
    where, parametres = filtre(periode, version)
    return premiere_ligne(
        sql.SQL("""
            SELECT count(*) AS nombre,
                   avg((decision = 'rejected')::int)::float AS taux_refus,
                   avg(probability) AS proba_moyenne,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS inference_p50_ms,
                   avg(application_ratio) AS couverture_dossier
            FROM predictions WHERE {where}
        """).format(where=where),
        parametres,
    )


@st.cache_data(ttl=30, show_spinner=False)
def serie_requetes(periode: tuple, pas: str) -> pd.DataFrame:
    where, parametres = filtre(periode)
    return lire(
        sql.SQL("""
            SELECT date_trunc({pas}, occurred_at) AS instant,
                   CASE WHEN status_code >= 500 THEN '5xx'
                        WHEN status_code >= 400 THEN '4xx'
                        ELSE '2xx' END AS classe,
                   count(*) AS appels
            FROM requests WHERE {where}
            GROUP BY 1, 2 ORDER BY 1
        """).format(pas=sql.Literal(pas), where=where),
        parametres,
    )


@st.cache_data(ttl=30, show_spinner=False)
def serie_latence(periode: tuple, pas: str) -> pd.DataFrame:
    where, parametres = filtre(periode)
    return lire(
        sql.SQL("""
            SELECT date_trunc({pas}, occurred_at) AS instant,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95
            FROM requests
            WHERE {where} AND path = '/predict' AND status_code < 400
            GROUP BY 1 ORDER BY 1
        """).format(pas=sql.Literal(pas), where=where),
        parametres,
    )


@st.cache_data(ttl=30, show_spinner=False)
def serie_predictions(periode: tuple, version: str | None, pas: str) -> pd.DataFrame:
    where, parametres = filtre(periode, version)
    return lire(
        sql.SQL("""
            SELECT date_trunc({pas}, occurred_at) AS instant,
                   count(*) AS predictions,
                   avg((decision = 'rejected')::int)::float AS taux_refus,
                   avg(application_ratio) AS couverture_dossier,
                   avg(history_ratio) AS couverture_historique
            FROM predictions WHERE {where}
            GROUP BY 1 ORDER BY 1
        """).format(pas=sql.Literal(pas), where=where),
        parametres,
    )


@st.cache_data(ttl=30, show_spinner=False)
def erreurs(periode: tuple) -> pd.DataFrame:
    where, parametres = filtre(periode)
    return lire(
        sql.SQL("""
            SELECT path AS route, status_code AS statut, count(*) AS appels
            FROM requests WHERE {where} AND status_code >= 400
            GROUP BY 1, 2 ORDER BY 3 DESC
        """).format(where=where),
        parametres,
    )


@st.cache_data(ttl=30, show_spinner=False)
def valeurs_production(periode: tuple, version: str | None, noms: tuple[str, ...]):
    where, parametres = filtre(periode, version)
    colonnes = sql.SQL(", ").join(
        sql.SQL("(features->>{})::float AS {}").format(sql.Literal(n), sql.Identifier(n))
        for n in noms
    )
    requete = sql.SQL("SELECT probability, {colonnes} FROM predictions WHERE {where}").format(
        colonnes=colonnes, where=where
    )
    return lire(requete, parametres).astype("float64")


def carte(colonne, titre: str, valeur: str, aide: str) -> None:
    colonne.metric(titre, valeur, help=aide, border=True)


def entier(valeur) -> str:
    return f"{int(valeur):,}".replace(",", "\u202f")


def format_ms(valeur) -> str:
    return "—" if valeur is None else f"{indicateurs.decimal(valeur, 1)} ms"


def format_pct(valeur) -> str:
    return "—" if valeur is None else indicateurs.pourcentage(valeur)


def afficher(graphique) -> None:
    st.altair_chart(graphique.configure(locale=LOCALE_FR), width="stretch")


def graphique_activite(serie: pd.DataFrame) -> alt.Chart:
    couleurs = alt.Scale(domain=["2xx", "4xx", "5xx"], range=["#4c78a8", "#f2a541", "#d64545"])
    return (
        alt.Chart(serie)
        .mark_bar()
        .encode(
            x=alt.X("instant:T", title=None, axis=AXE_TEMPS),
            y=alt.Y("sum(appels):Q", title="Appels"),
            color=alt.Color("classe:N", scale=couleurs, title="Statut"),
            tooltip=["instant:T", "classe:N", "sum(appels):Q"],
        )
        .properties(height=260)
    )


def graphique_latence(serie: pd.DataFrame) -> alt.Chart:
    centiles = serie.melt(
        id_vars="instant", value_vars=["p50", "p95"], var_name="centile", value_name="ms"
    )
    return (
        alt.Chart(centiles)
        .mark_line(point=True)
        .encode(
            x=alt.X("instant:T", title=None, axis=AXE_TEMPS),
            y=alt.Y("ms:Q", title="Durée de POST /predict (ms)"),
            color=alt.Color("centile:N", title=None),
            tooltip=["instant:T", "centile:N", alt.Tooltip("ms:Q", format=".2f")],
        )
        .properties(height=260)
    )


def graphique_scores(probabilites: pd.Series) -> alt.LayerChart:
    donnees = pd.DataFrame({"probabilite": probabilites.clip(0, 1)})
    histogramme = (
        alt.Chart(donnees)
        .mark_bar(color="#4c78a8")
        .encode(
            x=alt.X("probabilite:Q", bin=alt.Bin(step=0.02), title="Probabilité de défaut"),
            y=alt.Y("count():Q", title="Dossiers"),
        )
    )
    seuil = (
        alt.Chart(pd.DataFrame({"seuil": [SEUIL_DECISION]}))
        .mark_rule(color="#d64545", strokeDash=[6, 4], size=2)
        .encode(x="seuil:Q")
    )
    return (histogramme + seuil).properties(height=260)


def graphique_refus(serie: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(serie)
        .mark_line(point=True, color="#d64545")
        .encode(
            x=alt.X("instant:T", title=None, axis=AXE_TEMPS),
            y=alt.Y("taux_refus:Q", title="Taux de refus", axis=alt.Axis(format=".0%")),
            tooltip=["instant:T", alt.Tooltip("taux_refus:Q", format=".1%"), "predictions:Q"],
        )
        .properties(height=220)
    )


def graphique_derive(tableau: pd.DataFrame) -> alt.LayerChart:
    couleurs = alt.Scale(
        domain=["stable", "modérée", "significative"], range=["#59a14f", "#f2a541", "#d64545"]
    )
    barres = (
        alt.Chart(tableau)
        .mark_bar()
        .encode(
            x=alt.X("psi:Q", title="PSI"),
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("bande:N", scale=couleurs, title="Dérive"),
            tooltip=["feature:N", alt.Tooltip("psi:Q", format=".3f"), "bande:N"],
        )
    )
    seuils = (
        alt.Chart(pd.DataFrame({"seuil": [indicateurs.PSI_STABLE, indicateurs.PSI_SIGNIFICATIF]}))
        .mark_rule(strokeDash=[4, 4], color="#666")
        .encode(x="seuil:Q")
    )
    return (barres + seuils).properties(height=520)


def afficher_derive(derives: list[dict], profil: dict) -> None:
    tableau = pd.DataFrame(derives)
    afficher(graphique_derive(tableau))
    lignes_reference = entier(profil["lignes"])
    st.caption(
        f"PSI par déciles contre {lignes_reference} dossiers du jeu d'entraînement ; "
        f"les {len(derives)} features suivies portent "
        f"{indicateurs.pourcentage(profil['part_du_gain'])} du gain du "
        "modèle. Lignes pointillées : 0,10 (dérive modérée) et 0,25 (significative)."
    )
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
            "importance": st.column_config.NumberColumn("Gain dans le modèle", format="localized"),
        },
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
    st.set_page_config(page_title="Monitoring — scoring de crédit", layout="wide")
    st.title("Suivi du modèle de scoring en production")

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
        st.caption(f"Base : {DSN.rsplit('@', 1)[-1]}")

    version = None if choix_version == "Toutes" else choix_version
    pas = indicateurs.pas_temporel(duree)

    profil = json.loads(PROFIL.read_text(encoding="utf-8"))
    noms = tuple(profil["features"])

    requetes = resume_requetes(periode)
    predictions = resume_predictions(periode, version)
    production = valeurs_production(periode, version, noms)
    derives = indicateurs.derive(profil, {n: production[n].to_numpy() for n in noms})

    for niveau, texte in indicateurs.alertes(requetes, predictions, derives):
        getattr(st, niveau)(texte)

    appels = requetes["appels"]
    taux_erreur = (requetes["erreurs_4xx"] + requetes["erreurs_5xx"]) / appels if appels else None
    psi_lisible = predictions["nombre"] >= indicateurs.MIN_PREDICTIONS_PSI
    derivees = sum(d["bande"] == "significative" for d in derives)

    ligne = st.columns(6)
    carte(ligne[0], "Appels", entier(appels), "Appels des routes suivies.")
    carte(ligne[1], "Taux d'erreur", format_pct(taux_erreur), "Réponses 4xx et 5xx.")
    carte(ligne[2], "Latence p95", format_ms(requetes["latence_p95_ms"]),
          "Durée côté serveur des appels /predict, hors réseau.")
    carte(ligne[3], "Prédictions", entier(predictions["nombre"]),
          "Dossiers scorés, lots compris.")
    carte(ligne[4], "Taux de refus", format_pct(predictions["taux_refus"]),
          "Part des dossiers au-dessus du seuil de 0,10.")
    carte(ligne[5], "Dérive", f"{derivees} / {len(derives)}" if psi_lisible else "—",
          "Features suivies dont le PSI dépasse 0,25.")

    activite, scores, latence, derive_tab = st.tabs(
        ["Activité et erreurs", "Scores et décisions", "Latence", "Dérive des données"]
    )

    serie_req = serie_requetes(periode, pas)
    serie_pred = serie_predictions(periode, version, pas)

    with activite:
        if serie_req.empty:
            st.info("Aucun appel sur la période.")
        else:
            afficher(graphique_activite(serie_req))
            detail = erreurs(periode)
            st.subheader("Erreurs par route")
            if detail.empty:
                st.write("Aucune erreur sur la période.")
            else:
                st.dataframe(detail, hide_index=True, width="stretch")

    with scores:
        if production.empty:
            st.info("Aucune prédiction sur la période.")
        else:
            gauche, droite = st.columns(2)
            with gauche:
                st.subheader("Distribution des scores")
                afficher(graphique_scores(production["probability"]))
                st.caption("La ligne rouge marque le seuil de décision (0,10).")
            with droite:
                st.subheader("Taux de refus")
                afficher(graphique_refus(serie_pred))
                st.caption(
                    "Couverture moyenne des dossiers : "
                    f"{format_pct(predictions['couverture_dossier'])} des features de demande."
                )

    with latence:
        serie_lat = serie_latence(periode, pas)
        if serie_lat.empty:
            st.info("Aucun appel réussi à /predict sur la période.")
        else:
            afficher(graphique_latence(serie_lat))
            st.caption(
                "Durée mesurée par l'API, de la réception de la requête à l'envoi de la "
                f"réponse. Inférence seule, médiane : {format_ms(predictions['inference_p50_ms'])}."
            )

    with derive_tab:
        if psi_lisible:
            afficher_derive(derives, profil)
        else:
            st.info(
                f"{predictions['nombre']} prédictions sur la période : le PSI n'est calculé "
                f"qu'à partir de {indicateurs.MIN_PREDICTIONS_PSI}. Élargir la fenêtre."
            )

    with st.expander("Comment lire ce tableau de bord"):
        st.markdown(
            "- **Taux de refus** : c'est l'effet visible par le métier. Un écart durable "
            "de plus de 20 % par rapport à la semaine précédente justifie une analyse.\n"
            "- **Dérive** : le PSI localise la cause. Vérifier d'abord la couverture des "
            "dossiers : des dossiers plus incomplets changent les décisions sans que le "
            "modèle soit en cause.\n"
            "- **Erreurs** : une hausse des 422 signale un appelant qui a changé son format ; "
            "toute 500 est un incident.\n"
            "- Le réentraînement se décide sur une baisse de performance mesurée, pas sur un "
            "PSI seul (voir `notebooks/07_data_drift.ipynb`)."
        )


if __name__ == "__main__":
    main()
