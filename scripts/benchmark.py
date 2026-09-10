"""Mesure la latence de l'API : étape par étape en processus, ou de bout en bout en HTTP.

    uv run python scripts/benchmark.py etapes
    uv run python scripts/benchmark.py profil [--top 15] [--dump profil.prof]
    uv run python scripts/benchmark.py http http://127.0.0.1:8000 [--concurrence 4]

`--sortie fichier.json` enregistre les résultats avec le contexte de la mesure.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXEMPLE = PROJECT_ROOT / "api" / "exemple_dossier_refuse.json"
TAILLE_LOT = 200


def percentiles(durees_ms: list[float]) -> dict[str, float]:
    ordonnees = sorted(durees_ms)

    def rang(q: float) -> float:
        return ordonnees[min(len(ordonnees) - 1, int(q * len(ordonnees)))]

    return {
        "n": len(ordonnees),
        "p50": round(statistics.median(ordonnees), 4),
        "p95": round(rang(0.95), 4),
        "p99": round(rang(0.99), 4),
        "moyenne": round(statistics.fmean(ordonnees), 4),
    }


def chronometrer(fonction, repetitions: int, echauffement: int = 50) -> dict[str, float]:
    for _ in range(echauffement):
        fonction()
    durees = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        fonction()
        durees.append((time.perf_counter() - debut) * 1000)
    return percentiles(durees)


def charger_contexte():
    from api.main import _to_response, _validate
    from api.model import ScoringModel
    from api.schemas import PredictionRequest

    modele = ScoringModel.load()
    brut = json.dumps({"features": json.loads(EXEMPLE.read_text(encoding="utf-8"))}).encode()
    features = json.loads(brut)["features"]

    def route():
        requete = PredictionRequest.model_validate_json(brut)
        couverture = _validate(modele, requete.features)
        prediction = modele.predict([requete.features], [couverture])[0]
        return _to_response(prediction, modele).model_dump_json()

    return modele, brut, features, route, _validate, PredictionRequest


def mesurer_etapes(repetitions: int) -> dict:
    modele, brut, features, route, valider, schema = charger_contexte()
    lot = [features] * TAILLE_LOT

    resultats = {
        "validation_pydantic": chronometrer(lambda: schema.model_validate_json(brut), repetitions),
        "controle_contrat": chronometrer(lambda: valider(modele, features), repetitions),
        "inference_unitaire": chronometrer(lambda: modele.predict([features]), repetitions),
        "route_sans_http": chronometrer(route, repetitions),
    }
    par_lot = chronometrer(lambda: modele.predict(lot), max(20, repetitions // 20), echauffement=5)
    resultats["inference_par_dossier_en_lot"] = {
        k: (round(v / TAILLE_LOT, 4) if k != "n" else v) for k, v in par_lot.items()
    }
    return resultats


def profiler(repetitions: int, top: int, dump: Path | None) -> str:
    *_, route, _, _ = charger_contexte()
    for _ in range(50):
        route()
    profil = cProfile.Profile()
    profil.enable()
    for _ in range(repetitions):
        route()
    profil.disable()
    if dump:
        profil.dump_stats(dump)
    sortie = io.StringIO()
    pstats.Stats(profil, stream=sortie).sort_stats("tottime").print_stats(top)
    return sortie.getvalue()


def mesurer_http(url: str, repetitions: int, concurrence: int) -> dict:
    import httpx

    corps = json.dumps({"features": json.loads(EXEMPLE.read_text(encoding="utf-8"))})
    entetes = {"Content-Type": "application/json"}

    def serie(n: int) -> list[float]:
        durees = []
        with httpx.Client(base_url=url, timeout=30) as client:
            for _ in range(10):
                client.post("/predict", content=corps, headers=entetes).raise_for_status()
            for _ in range(n):
                debut = time.perf_counter()
                client.post("/predict", content=corps, headers=entetes).raise_for_status()
                durees.append((time.perf_counter() - debut) * 1000)
        return durees

    resultats = {"sequentiel": percentiles(serie(repetitions))}
    if concurrence > 1:
        debut = time.perf_counter()
        with ThreadPoolExecutor(concurrence) as pool:
            series = list(pool.map(serie, [repetitions // concurrence] * concurrence))
        duree = time.perf_counter() - debut
        toutes = [d for s in series for d in s]
        resultats[f"concurrence_{concurrence}"] = {
            **percentiles(toutes),
            "debit_req_s": round(len(toutes) / duree, 1),
        }
    return resultats


def contexte_de_mesure() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "inconnu"
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit,
        "plateforme": platform.platform(),
        "python": platform.python_version(),
    }


def afficher(resultats: dict) -> None:
    largeur = max(len(nom) for nom in resultats)
    print(f"{'':{largeur}}  {'p50':>9} {'p95':>9} {'p99':>9}  (ms)")
    for nom, stats in resultats.items():
        extra = f"   {stats['debit_req_s']} req/s" if "debit_req_s" in stats else ""
        print(f"{nom:{largeur}}  {stats['p50']:9.3f} {stats['p95']:9.3f} {stats['p99']:9.3f}{extra}")


def main() -> int:
    parseur = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parseur.add_argument("mode", choices=["etapes", "profil", "http"])
    parseur.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
    parseur.add_argument("--repetitions", type=int, default=1000)
    parseur.add_argument("--concurrence", type=int, default=1)
    parseur.add_argument("--top", type=int, default=15)
    parseur.add_argument("--dump", type=Path)
    parseur.add_argument("--etiquette", default="")
    parseur.add_argument("--sortie", type=Path)
    args = parseur.parse_args()

    if args.mode == "profil":
        print(profiler(args.repetitions, args.top, args.dump))
        return 0

    if args.mode == "etapes":
        resultats = mesurer_etapes(args.repetitions)
    else:
        resultats = mesurer_http(args.url, args.repetitions, args.concurrence)
    afficher(resultats)

    if args.sortie:
        args.sortie.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "etiquette": args.etiquette,
            "mode": args.mode,
            "cible": args.url if args.mode == "http" else "processus",
            **contexte_de_mesure(),
            "resultats": resultats,
        }
        args.sortie.write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        print(f"\n-> {args.sortie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
