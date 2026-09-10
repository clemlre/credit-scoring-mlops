"""Test de charge de plusieurs conteneurs de l'API avec oha, en alternant les cibles.

oha tourne dans un conteneur qui partage l'espace réseau de la cible
(`--network container:<nom>`) : ni le générateur Python, ni le réseau de Docker
Desktop n'entrent dans la mesure, et le générateur n'est pas soumis au quota CPU de
la cible.

    uv run python scripts/tester_charge.py bench-avant bench-final \
        --connexions 8 32 --duree 15s --tours 3 --sortie docs/perf/charge.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

from benchmark import EXEMPLE, contexte_de_mesure

OHA = "ghcr.io/hatoo/oha:latest"


def ecrire_corps(dossier: Path, lot: int) -> str:
    """Écrit `corps.json` dans `dossier` et renvoie la route à viser."""
    features = json.loads(EXEMPLE.read_text(encoding="utf-8"))
    if lot:
        route, corps = "/predict/batch", {"items": [{"features": features}] * lot}
    else:
        route, corps = "/predict", {"features": features}
    (dossier / "corps.json").write_text(json.dumps(corps), encoding="utf-8")
    return route


def tir(conteneur: str, connexions: int, duree: str, dossier: Path, route: str) -> dict:
    sortie = subprocess.run(
        [
            "docker", "run", "--rm", "--network", f"container:{conteneur}",
            "-v", f"{dossier}:/corps:ro", OHA,
            "-z", duree, "-c", str(connexions), "-m", "POST",
            "-H", "Content-Type: application/json", "-D", "/corps/corps.json",
            "--no-tui", "--output-format", "json", f"http://127.0.0.1:8000{route}",
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    rapport = json.loads(sortie)
    centiles = rapport["latencyPercentiles"]
    return {
        "debit_req_s": round(rapport["summary"]["requestsPerSec"], 1),
        "taux_succes": rapport["summary"]["successRate"],
        "p50": round(centiles["p50"] * 1000, 2),
        "p95": round(centiles["p95"] * 1000, 2),
        "p99": round(centiles["p99"] * 1000, 2),
    }


def main() -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("conteneurs", nargs="+")
    parseur.add_argument("--connexions", type=int, nargs="+", default=[8, 32])
    parseur.add_argument("--duree", default="15s")
    parseur.add_argument("--tours", type=int, default=3)
    parseur.add_argument("--lot", type=int, default=0)
    parseur.add_argument("--sortie", type=Path)
    args = parseur.parse_args()

    with tempfile.TemporaryDirectory() as temporaire:
        dossier = Path(temporaire)
        route = ecrire_corps(dossier, args.lot)

        tours = {c: {n: [] for n in args.connexions} for c in args.conteneurs}
        for tour in range(1, args.tours + 1):
            for connexions in args.connexions:
                for conteneur in args.conteneurs:
                    mesure = tir(conteneur, connexions, args.duree, dossier, route)
                    tours[conteneur][connexions].append(mesure)
                    centiles = "  ".join(f"{c} {mesure[c]:6.2f}" for c in ("p50", "p95", "p99"))
                    debit = mesure["debit_req_s"]
                    entete = f"tour {tour} c={connexions:<3} {conteneur:14}"
                    print(f"{entete} {debit:7.1f} req/s {centiles} ms")

    synthese = {
        conteneur: {
            str(connexions): {
                cle: round(statistics.median(m[cle] for m in mesures), 2) for cle in mesures[0]
            }
            for connexions, mesures in par_charge.items()
        }
        for conteneur, par_charge in tours.items()
    }
    print(json.dumps(synthese, indent=2))

    if args.sortie:
        document = {
            **contexte_de_mesure(),
            "protocole": {
                "outil": OHA,
                "route": route,
                "dossiers_par_requete": args.lot or 1,
                "duree": args.duree,
                "tours": args.tours,
            },
            "synthese_mediane_des_tours": synthese,
            "tours": tours,
        }
        args.sortie.write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
