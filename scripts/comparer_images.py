"""Compare deux conteneurs de l'API en alternant les mesures, pour lisser le bruit
de la machine.

Chaque tour mesure, pour chaque conteneur, une phase séquentielle puis une phase
concurrente. Pour chaque phase on relève la latence vue du client et la durée
côté serveur, lue dans les journaux JSON du conteneur (`event: request`) : cette
seconde mesure ne dépend pas du réseau de Docker Desktop.

    uv run python scripts/comparer_images.py avant=bench-avant:8010 apres=bench-apres:8011 \
        --tours 3 --sortie docs/perf/comparaison.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from benchmark import contexte_de_mesure, mesurer_http, percentiles


def durees_serveur(conteneur: str, nombre: int) -> list[float]:
    """Durées des `nombre` derniers appels /predict réussis, lues dans les journaux.

    On lit les dernières lignes plutôt qu'une fenêtre horaire : l'horloge de la VM
    Docker peut être décalée de celle de l'hôte."""
    journaux = subprocess.run(
        ["docker", "logs", "--tail", str(4 * nombre), conteneur],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    durees = []
    for ligne in journaux.splitlines():
        if not ligne.startswith('{"event": "request"'):
            continue
        evenement = json.loads(ligne)
        if evenement["path"] == "/predict" and evenement["status_code"] == 200:
            durees.append(evenement["duration_ms"])
    return durees[-nombre:]


def phase(conteneur: str, url: str, repetitions: int, concurrence: int) -> dict:
    client = mesurer_http(url, repetitions, concurrence, seulement_concurrent=concurrence > 1)
    time.sleep(1)
    cle = "sequentiel" if concurrence == 1 else f"concurrence_{concurrence}"
    return {"client": client[cle], "serveur": percentiles(durees_serveur(conteneur, repetitions))}


def mediane_des_tours(tours: list[dict]) -> dict:
    return {
        cle: round(statistics.median(t[cle] for t in tours), 3)
        for cle in tours[0]
        if cle != "n"
    }


def main() -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("cibles", nargs=2, help="nom=conteneur:port")
    parseur.add_argument("--tours", type=int, default=3)
    parseur.add_argument("--repetitions", type=int, default=1000)
    parseur.add_argument("--concurrence", type=int, default=8)
    parseur.add_argument("--sortie", type=Path)
    args = parseur.parse_args()

    cibles = {}
    for cible in args.cibles:
        nom, reste = cible.split("=")
        conteneur, port = reste.split(":")
        cibles[nom] = (conteneur, f"http://127.0.0.1:{port}")

    brut = {nom: {"sequentiel": [], "concurrent": []} for nom in cibles}
    for tour in range(1, args.tours + 1):
        for nom, (conteneur, url) in cibles.items():
            sequentiel = phase(conteneur, url, args.repetitions, 1)
            concurrent = phase(conteneur, url, args.repetitions, args.concurrence)
            brut[nom]["sequentiel"].append(sequentiel)
            brut[nom]["concurrent"].append(concurrent)
            print(
                f"tour {tour} {nom:6} séquentiel p50 client {sequentiel['client']['p50']:6.2f} "
                f"serveur {sequentiel['serveur']['p50']:6.2f} | concurrent "
                f"{concurrent['client']['debit_req_s']:6.1f} req/s, "
                f"p95 client {concurrent['client']['p95']:6.2f} ms"
            )

    synthese = {
        nom: {
            charge: {
                "client": mediane_des_tours([m["client"] for m in mesures]),
                "serveur": mediane_des_tours([m["serveur"] for m in mesures]),
            }
            for charge, mesures in phases.items()
        }
        for nom, phases in brut.items()
    }
    print(json.dumps(synthese, indent=2, ensure_ascii=False))

    if args.sortie:
        document = {
            **contexte_de_mesure(),
            "protocole": {
                "tours": args.tours,
                "repetitions": args.repetitions,
                "concurrence": args.concurrence,
                "cibles": {nom: conteneur for nom, (conteneur, _) in cibles.items()},
            },
            "synthese_mediane_des_tours": synthese,
            "tours": brut,
        }
        args.sortie.parent.mkdir(parents=True, exist_ok=True)
        args.sortie.write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
