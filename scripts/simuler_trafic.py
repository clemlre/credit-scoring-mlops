"""Envoie du trafic à l'API pour alimenter le journal des prédictions.

À quoi ça sert
--------------
Une solution de stockage vide ne se démontre pas, et une analyse de dérive sans
données de production n'a rien à comparer. Ce script fabrique le « trafic de
production » dont le monitoring a besoin.

D'où viennent les dossiers
--------------------------
Par défaut, de **vrais dossiers** du jeu de la Partie 1 (`feature_dataset.parquet`),
qui vit hors du dépôt. C'est important : des valeurs inventées suivraient une
distribution inventée, et l'analyse de dérive comparerait alors deux fictions. Si le
parquet est introuvable, on retombe sur des dossiers synthétiques — utilisables pour
une démonstration, mais **pas** pour conclure quoi que ce soit sur la dérive.

Aucune donnée client n'est écrite dans le dépôt : elle est lue depuis le disque et
envoyée à l'API, rien de plus.

Usage
-----
    python scripts/simuler_trafic.py                          # 100 dossiers, port 8000
    python scripts/simuler_trafic.py --url http://127.0.0.1:8001 --nombre 500
    python scripts/simuler_trafic.py --decalage 0.15          # dossiers volontairement décalés
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT = 30
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Emplacement par défaut du projet Partie 1, surchargeable par l'environnement.
P6_DEFAUT = r"C:\Users\ClementLoire\ObsidianVault_MCP_enabled\05 - Cours\P6 - Initiez-vous au MLOps 1-2"


def appeler(url: str, chemin: str, charge: dict | None = None) -> tuple[int, dict]:
    donnees = json.dumps(charge).encode() if charge is not None else None
    requete = urllib.request.Request(
        f"{url.rstrip('/')}{chemin}",
        data=donnees,
        headers={"Content-Type": "application/json"} if donnees else {},
        method="POST" if donnees else "GET",
    )
    try:
        with urllib.request.urlopen(requete, timeout=TIMEOUT) as reponse:
            return reponse.status, json.loads(reponse.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def dossiers_reels(noms_features: list[str], nombre: int) -> list[dict] | None:
    """Lit des dossiers réels dans le parquet de la Partie 1, ou None s'il est absent."""
    parquet = Path(os.environ.get("P6_PROJECT_ROOT", P6_DEFAUT)) / "output" / "feature_dataset.parquet"
    if not parquet.exists():
        return None

    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(parquet)
    # On tire dans les dossiers *sans* étiquette : ce sont ceux que le modèle n'a
    # jamais vus, donc les plus représentatifs d'un flux de production.
    candidats = frame[frame["TARGET"].isna()]
    if len(candidats) < nombre:
        candidats = frame
    echantillon = candidats.sample(n=min(nombre, len(candidats)), random_state=42)

    X = echantillon.drop(columns=[c for c in ("TARGET", "SK_ID_CURR") if c in echantillon.columns])
    X = X.drop(columns=X.select_dtypes(include="object").columns)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X[[c for c in noms_features if c in X.columns]]

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
    """Décale volontairement les scores externes, pour fabriquer de la dérive.

    Sert à vérifier qu'un tableau de bord de monitoring **détecte** bien un
    changement : un détecteur qu'on n'a jamais vu se déclencher ne prouve rien.
    """
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

    envoyes, refuses, acceptes = 0, 0, 0
    for depart in range(0, len(dossiers), arguments.lot):
        tranche = dossiers[depart : depart + arguments.lot]
        statut, corps = appeler(
            arguments.url, "/predict/batch", {"items": [{"features": d} for d in tranche]}
        )
        if statut != 200:
            print(f"  lot refusé (statut {statut}) : {corps.get('detail', '')[:160]}")
            continue
        envoyes += len(tranche)
        for prediction in corps["predictions"]:
            if prediction["decision"] == "accepted":
                acceptes += 1
            else:
                refuses += 1

    print(f"\n{envoyes} prédictions journalisées — {acceptes} acceptées, {refuses} refusées.")
    return 0 if envoyes else 1


if __name__ == "__main__":
    raise SystemExit(main())
