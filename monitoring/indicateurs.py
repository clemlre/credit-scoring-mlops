"""Calculs du tableau de bord de monitoring, sans dépendance à Streamlit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

FENETRES: dict[str, timedelta | None] = {
    "Dernière heure": timedelta(hours=1),
    "24 heures": timedelta(days=1),
    "7 jours": timedelta(days=7),
    "30 jours": timedelta(days=30),
    "Tout l'historique": None,
}

# Bandes de lecture du PSI, les mêmes que dans notebooks/07_data_drift.ipynb.
PSI_STABLE = 0.10
PSI_SIGNIFICATIF = 0.25

# Sous ce volume, un PSI sur 10 classes n'est plus interprétable.
MIN_PREDICTIONS_PSI = 500

SEUIL_TAUX_4XX = 0.05
SEUIL_LATENCE_P95_MS = 100.0

# Écart relatif toléré entre le taux de refus observé et le taux attendu du modèle.
SEUIL_ECART_TAUX_REFUS = 0.20

EPSILON = 1e-4


def pourcentage(valeur: float) -> str:
    return f"{valeur:.1%}".replace(".", ",").replace("%", " %")


def decimal(valeur: float, chiffres: int = 2) -> str:
    return f"{valeur:.{chiffres}f}".replace(".", ",")


def entier(valeur) -> str:
    return f"{int(valeur):,}".replace(",", " ")


def periode_precise(du: str, au: str | None, maintenant: datetime) -> tuple[datetime, datetime]:
    """Période lue dans l'URL (?du=...&au=...), en ISO 8601. Sans fuseau : UTC."""

    def lire(texte: str) -> datetime:
        instant = datetime.fromisoformat(texte)
        return instant if instant.tzinfo else instant.replace(tzinfo=UTC)

    debut = lire(du)
    fin = lire(au) if au else maintenant
    if debut >= fin:
        raise ValueError("le début de la période doit précéder sa fin")
    return debut, fin


def pas_temporel(fenetre: timedelta | None) -> str:
    """Unité d'agrégation des séries (argument de date_trunc)."""
    if fenetre is not None and fenetre <= timedelta(hours=2):
        return "minute"
    if fenetre is not None and fenetre <= timedelta(days=3):
        return "hour"
    return "day"


def bornes_deciles(valeurs: np.ndarray) -> list[float]:
    """Bornes intérieures des déciles, dédoublonnées (features discrètes)."""
    valeurs = valeurs[~np.isnan(valeurs)]
    return np.unique(np.quantile(valeurs, np.linspace(0, 1, 11)[1:-1])).tolist()


def repartition(valeurs: np.ndarray, bornes: list[float]) -> np.ndarray:
    """Part des valeurs non manquantes dans chaque classe définie par `bornes`."""
    valeurs = valeurs[~np.isnan(valeurs)]
    if len(valeurs) == 0:
        return np.zeros(len(bornes) + 1)
    classes = np.searchsorted(bornes, valeurs, side="right")
    return np.bincount(classes, minlength=len(bornes) + 1) / len(valeurs)


def psi(reference: np.ndarray, production: np.ndarray) -> float:
    """Population Stability Index entre deux répartitions sur les mêmes classes."""
    r = np.clip(np.asarray(reference, dtype=float), EPSILON, None)
    p = np.clip(np.asarray(production, dtype=float), EPSILON, None)
    return float(np.sum((p - r) * np.log(p / r)))


def bande(valeur: float) -> str:
    if valeur < PSI_STABLE:
        return "stable"
    if valeur < PSI_SIGNIFICATIF:
        return "modérée"
    return "significative"


def profil_feature(valeurs: np.ndarray) -> dict:
    bornes = bornes_deciles(valeurs)
    return {
        "bornes": bornes,
        "proportions": repartition(valeurs, bornes).round(6).tolist(),
        "taux_manquant": round(float(np.isnan(valeurs).mean()), 6),
    }


def derive(profil: dict, production: dict[str, np.ndarray]) -> list[dict]:
    """PSI et écart de taux de manquants pour chaque feature du profil de référence."""
    lignes = []
    for nom, ref in profil["features"].items():
        valeurs = production.get(nom, np.array([]))
        indice = round(psi(ref["proportions"], repartition(valeurs, ref["bornes"])), 4)
        manquant = round(float(np.isnan(valeurs).mean()), 4) if len(valeurs) else None
        lignes.append(
            {
                "feature": nom,
                "psi": indice,
                "bande": bande(indice),
                "manquant_reference": ref["taux_manquant"],
                "manquant_production": manquant,
                "importance": ref["importance"],
            }
        )
    return sorted(lignes, key=lambda ligne: ligne["psi"], reverse=True)


# Les trois questions du tableau de bord, dans l'ordre de lecture. Chaque état est un
# couple (niveau, texte) ; le niveau vaut ok, info, warning ou error.


def etat_decisions(predictions: dict, attendu: float | None) -> tuple[str, str]:
    """1. Le modèle décide-t-il comme avant ? Le taux de refus contre le taux attendu."""
    nombre, taux = predictions["nombre"], predictions["taux_refus"]
    if not nombre:
        return "info", "Aucune prédiction sur la période."
    if attendu is None:
        return "info", (
            f"Taux de refus {pourcentage(taux)} ; aucun taux attendu dans les métadonnées "
            "du modèle pour le comparer."
        )
    ecart = (taux - attendu) / attendu
    if abs(ecart) > SEUIL_ECART_TAUX_REFUS:
        signe = f"{ecart:+.0%}".replace("%", " %")
        return "warning", (
            f"Taux de refus {pourcentage(taux)} contre {pourcentage(attendu)} attendu "
            f"({signe} d'écart relatif)."
        )
    tolerance = f"{SEUIL_ECART_TAUX_REFUS:.0%}".replace("%", " %")
    return "ok", (
        f"Taux de refus {pourcentage(taux)}, cohérent avec les {pourcentage(attendu)} "
        f"attendus (tolérance ± {tolerance})."
    )


def etat_donnees(derives: list[dict], nombre: int) -> tuple[str, str]:
    """2. Les données ont-elles changé ? Le PSI des features suivies."""
    if nombre < MIN_PREDICTIONS_PSI:
        return "info", (
            f"{entier(nombre)} prédictions : il en faut au moins {MIN_PREDICTIONS_PSI} pour "
            "lire le PSI. Élargir la fenêtre."
        )
    significatives = [d["feature"] for d in derives if d["bande"] == "significative"]
    if significatives:
        return "warning", (
            f"Dérive significative (PSI ≥ {decimal(PSI_SIGNIFICATIF)}) sur "
            f"{len(significatives)} feature(s) : {', '.join(significatives)}."
        )
    moderees = sum(d["bande"] == "modérée" for d in derives)
    if moderees:
        return "ok", (
            f"Aucune dérive significative ; {moderees} feature(s) en dérive modérée "
            f"sur {len(derives)} suivies."
        )
    return "ok", f"Aucune dérive sur les {len(derives)} features suivies."


def etat_service(requetes: dict) -> tuple[str, str]:
    """3. Le service tient-il ? Erreurs internes, appels refusés, latence."""
    appels = requetes["appels"]
    if not appels:
        return "info", "Aucun appel sur la période."
    problemes, niveau = [], "ok"
    if requetes["erreurs_5xx"]:
        problemes.append(f"{entier(requetes['erreurs_5xx'])} erreur(s) interne(s) (5xx)")
        niveau = "error"
    taux_4xx = requetes["erreurs_4xx"] / appels
    if taux_4xx > SEUIL_TAUX_4XX:
        problemes.append(
            f"{pourcentage(taux_4xx)} d'appels refusés (4xx), seuil {pourcentage(SEUIL_TAUX_4XX)}"
        )
        niveau = "warning" if niveau == "ok" else niveau
    p95 = requetes["latence_p95_ms"]
    if p95 is not None and p95 > SEUIL_LATENCE_P95_MS:
        problemes.append(f"latence p95 {p95:.0f} ms, objectif {SEUIL_LATENCE_P95_MS:.0f} ms")
        niveau = "warning" if niveau == "ok" else niveau
    if problemes:
        return niveau, " ; ".join(problemes).capitalize() + "."
    taux_erreur = (requetes["erreurs_4xx"] + requetes["erreurs_5xx"]) / appels
    latence = "N/A" if p95 is None else f"{p95:.0f} ms"
    return "ok", (
        f"{entier(appels)} appels, {pourcentage(taux_erreur)} d'erreurs, latence p95 "
        f"{latence} (objectif {SEUIL_LATENCE_P95_MS:.0f} ms)."
    )


def metriques_modele(metadata: dict) -> dict:
    """Performances out-of-fold lues dans la carte d'identité du modèle (None si absentes)."""
    metriques = metadata.get("metrics", {})
    return {
        "auc": metriques.get("auc_oof"),
        "accuracy": metriques.get("accuracy_oof"),
        "precision": metriques.get("precision_oof"),
        "rappel": metriques.get("recall_oof"),
        "f1": metriques.get("f1_oof"),
        "taux_refus_attendu": metriques.get("rejection_rate_oof"),
        "accuracy_seuil_naif": metriques.get("accuracy_oof_threshold_0.5"),
        "rappel_seuil_naif": metriques.get("recall_oof_threshold_0.5"),
    }
