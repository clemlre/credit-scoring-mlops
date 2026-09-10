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

EPSILON = 1e-4


def pourcentage(valeur: float) -> str:
    return f"{valeur:.1%}".replace(".", ",").replace("%", " %")


def decimal(valeur: float, chiffres: int = 2) -> str:
    return f"{valeur:.{chiffres}f}".replace(".", ",")


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


def alertes(requetes: dict, predictions: dict, derives: list[dict]) -> list[tuple[str, str]]:
    """Messages (niveau, texte) à afficher en tête du tableau de bord."""
    messages = []
    if requetes["erreurs_5xx"]:
        messages.append(("error", f"{requetes['erreurs_5xx']} erreur(s) interne(s) (5xx)."))

    if requetes["appels"]:
        taux_4xx = requetes["erreurs_4xx"] / requetes["appels"]
        if taux_4xx > SEUIL_TAUX_4XX:
            texte = f"{pourcentage(taux_4xx)} des appels sont refusés (4xx) : "
            texte += "un appelant envoie peut-être des dossiers mal formés."
            messages.append(("warning", texte))

    p95 = requetes["latence_p95_ms"]
    if p95 is not None and p95 > SEUIL_LATENCE_P95_MS:
        texte = f"Latence p95 de {p95:.0f} ms, objectif {SEUIL_LATENCE_P95_MS:.0f} ms."
        messages.append(("warning", texte))

    if predictions["nombre"] < MIN_PREDICTIONS_PSI:
        texte = f"{predictions['nombre']} prédictions sur la période : il en faut au moins "
        texte += f"{MIN_PREDICTIONS_PSI} pour que le PSI soit interprétable."
        messages.append(("info", texte))
        return messages

    significatives = [d["feature"] for d in derives if d["bande"] == "significative"]
    if significatives:
        texte = f"Dérive significative (PSI ≥ {decimal(PSI_SIGNIFICATIF)}) sur "
        texte += f"{len(significatives)} feature(s) : {', '.join(significatives)}."
        messages.append(("warning", texte))
    return messages
