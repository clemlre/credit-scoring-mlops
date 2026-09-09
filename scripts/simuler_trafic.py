"""Envoie du trafic à l'API pour alimenter le journal des prédictions.

Les dossiers sont tirés du jeu de la Partie 1 (hors dépôt) ; à défaut, ils sont
synthétiques et ne permettent pas de conclure sur la dérive.

    python scripts/simuler_trafic.py --url http://127.0.0.1:8001 --nombre 500
    python scripts/simuler_trafic.py --decalage 0.15    # dérive volontaire
    python scripts/simuler_trafic.py --unitaire --par-seconde 5 --taux-erreur 0.02
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

from smoke_test import call as appeler


def dossiers_reels(noms_features: list[str], nombre: int) -> list[dict] | None:
    """Lit des dossiers réels dans le parquet de la Partie 1, ou None s'il est absent."""
    racine = os.environ.get("P6_PROJECT_ROOT")
    if not racine:
        return None
    parquet = Path(racine) / "output" / "feature_dataset.parquet"
    if not parquet.exists():
        return None

    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(parquet)
    # Dossiers *sans* étiquette : jamais vus par le modèle, donc les plus
    # représentatifs d'un flux de production.
    echantillon = frame[frame["TARGET"].isna()].sample(n=nombre, random_state=42)
    X = echantillon[noms_features].replace([np.inf, -np.inf], np.nan)

    return [
        {nom: (None if pd.isna(valeur) else float(valeur)) for nom, valeur in ligne.items()}
        for ligne in X.to_dict(orient="records")
    ]


def dossiers_synthetiques(features_dossier: list[str], nombre: int) -> list[dict]:
    """Repli sans données réelles : des dossiers valides mais inventés."""
    alea = random.Random(42)
    dossiers = []
    for _ in range(nombre):
        dossier = {}
        for nom in features_dossier:
            if nom.startswith("EXT_SOURCE"):
                dossier[nom] = round(alea.uniform(0.05, 0.95), 4)
            elif nom.startswith("FLAG_"):
                dossier[nom] = float(alea.randint(0, 1))
            elif nom.startswith("DAYS_") and not nom.endswith("_PERC"):
                dossier[nom] = float(-alea.randint(500, 20000))
            elif nom.startswith("AMT_"):
                dossier[nom] = round(alea.uniform(50_000, 900_000), 2)
            elif nom.startswith("CNT_"):
                dossier[nom] = float(alea.randint(0, 4))
            else:
                dossier[nom] = round(alea.uniform(0, 1), 4)
        dossiers.append(dossier)
    return dossiers


def decaler(dossiers: list[dict], intensite: float) -> list[dict]:
    """Décale les scores externes, pour vérifier que la dérive est bien détectée."""
    if intensite <= 0:
        return dossiers
    decales = []
    for dossier in dossiers:
        copie = dict(dossier)
        for nom, valeur in dossier.items():
            if nom.startswith("EXT_SOURCE") and valeur is not None:
                copie[nom] = max(0.0, min(1.0, valeur - intensite))
        decales.append(copie)
    return decales


def corrompre(dossiers: list[dict], taux: float) -> list[dict]:
    """Rend invalide une part des dossiers (montant négatif), pour faire apparaître
    des 422 dans le suivi du taux d'erreur."""
    alea = random.Random(7)
    return [
        {**dossier, "AMT_CREDIT": -1.0} if alea.random() < taux else dossier
        for dossier in dossiers
    ]


def envoyer_unitaire(url: str, dossiers: list[dict], par_seconde: float) -> tuple[int, int, int]:
    envoyes, acceptes, erreurs = 0, 0, 0
    intervalle = 1 / par_seconde if par_seconde else 0
    for dossier in dossiers:
        debut = time.perf_counter()
        statut, corps = appeler(url, "/predict", {"features": dossier})
        if statut == 200:
            envoyes += 1
            acceptes += corps["decision"] == "accepted"
        else:
            erreurs += 1
        time.sleep(max(0.0, intervalle - (time.perf_counter() - debut)))
    return envoyes, acceptes, erreurs


def envoyer_par_lots(url: str, dossiers: list[dict], taille: int) -> tuple[int, int, int]:
    envoyes, acceptes, erreurs = 0, 0, 0
    for depart in range(0, len(dossiers), taille):
        tranche = dossiers[depart : depart + taille]
        charge = {"items": [{"features": d} for d in tranche]}
        statut, corps = appeler(url, "/predict/batch", charge)
        if statut != 200:
            print(f"  lot refusé (statut {statut}) : {corps.get('detail', '')[:160]}")
            erreurs += 1
            continue
        envoyes += len(tranche)
        acceptes += sum(p["decision"] == "accepted" for p in corps["predictions"])
    return envoyes, acceptes, erreurs


def main() -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--url", default="http://127.0.0.1:8000", help="Base de l'API.")
    parseur.add_argument("--nombre", type=int, default=100, help="Nombre de dossiers à envoyer.")
    parseur.add_argument("--lot", type=int, default=25, help="Taille des lots envoyés.")
    parseur.add_argument(
        "--decalage",
        type=float,
        default=0.0,
        help="Décale les scores externes de cette valeur, pour simuler une dérive.",
    )
    parseur.add_argument("--unitaire", action="store_true", help="Un appel /predict par dossier.")
    parseur.add_argument("--par-seconde", type=float, default=0, help="Cadence en mode unitaire.")
    parseur.add_argument(
        "--taux-erreur", type=float, default=0.0, help="Part de dossiers invalides."
    )
    arguments = parseur.parse_args()

    statut, contrat = appeler(arguments.url, "/features")
    if statut != 200:
        print(f"L'API ne répond pas correctement sur /features (statut {statut}).", file=sys.stderr)
        return 1

    noms = contrat["application_features"] + contrat["history_features"]
    dossiers = dossiers_reels(noms, arguments.nombre)
    if dossiers is None:
        print("Parquet de la Partie 1 introuvable — repli sur des dossiers synthétiques.")
        print("  (utilisable pour une démonstration, pas pour conclure sur la dérive)")
        dossiers = dossiers_synthetiques(contrat["application_features"], arguments.nombre)
    else:
        print(f"{len(dossiers)} dossiers réels tirés du jeu de la Partie 1.")

    dossiers = decaler(dossiers, arguments.decalage)
    if arguments.decalage:
        print(f"Décalage appliqué aux scores externes : -{arguments.decalage}")

    dossiers = corrompre(dossiers, arguments.taux_erreur)

    if arguments.unitaire:
        envoyes, acceptes, erreurs = envoyer_unitaire(
            arguments.url, dossiers, arguments.par_seconde
        )
    else:
        envoyes, acceptes, erreurs = envoyer_par_lots(arguments.url, dossiers, arguments.lot)

    print(
        f"\n{envoyes} prédictions journalisées — {acceptes} acceptées, "
        f"{envoyes - acceptes} refusées ; {erreurs} appel(s) en erreur."
    )
    return 0 if envoyes else 1


if __name__ == "__main__":
    raise SystemExit(main())
