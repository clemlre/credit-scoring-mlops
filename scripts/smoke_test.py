"""Vérifie qu'une instance déjà déployée de l'API répond correctement.

À la différence de la suite pytest, qui teste le code, ce script teste un
**service en train de tourner** : le conteneur fraîchement construit en CI, ou
l'instance déployée après une mise en production. C'est ce qui distingue « le code
est bon » de « le service marche ».

Il ne dépend que de la bibliothèque standard, pour pouvoir s'exécuter n'importe où
sans installer l'environnement du projet.

Le dossier de test n'est pas codé en dur : le script interroge `GET /features` et
construit un dossier valide à partir du contrat que l'API déclare. Un changement de
contrat ne le périme donc pas.

Usage :
    python scripts/smoke_test.py                       # http://127.0.0.1:8000
    python scripts/smoke_test.py https://mon-api.example.com
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

TIMEOUT = 15
# Le conteneur doit charger 5 Mo de modèle avant de répondre : on lui laisse le temps.
STARTUP_ATTEMPTS = 30
STARTUP_DELAY = 2


class SmokeTestError(RuntimeError):
    pass


def call(base_url: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    """Appelle une route et renvoie (statut, corps décodé)."""
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # Un 422 ou un 503 sont des réponses légitimes à vérifier, pas des pannes.
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"detail": body.decode(errors="replace")}


def wait_until_ready(base_url: str) -> None:
    """Attend que le service ait fini de charger son modèle."""
    for attempt in range(1, STARTUP_ATTEMPTS + 1):
        try:
            status, body = call(base_url, "/health")
            if status == 200 and body.get("model_loaded"):
                print(f"  service prêt après {attempt} tentative(s) — modèle v{body['model_version']}")
                return
            print(f"  tentative {attempt}/{STARTUP_ATTEMPTS} : statut {status}")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            print(f"  tentative {attempt}/{STARTUP_ATTEMPTS} : pas encore joignable ({exc.__class__.__name__})")
        time.sleep(STARTUP_DELAY)
    raise SmokeTestError("le service n'est pas devenu disponible dans le temps imparti")


def build_application_file(application_features: list[str]) -> dict[str, float]:
    """Construit un dossier de demande respectant les plages de validité de l'API."""
    values: dict[str, float] = {}
    for name in application_features:
        if name.startswith("EXT_SOURCE"):
            values[name] = 0.5
        elif name.startswith("FLAG_"):
            values[name] = 0.0
        elif name.startswith("DAYS_") and not name.endswith("_PERC"):
            values[name] = -5000.0
        elif name.startswith("AMT_"):
            values[name] = 100000.0
        elif name.startswith("CNT_"):
            values[name] = 1.0
        else:
            values[name] = 0.0
    return values


def main(base_url: str) -> int:
    print(f"Test de fumée sur {base_url}\n")

    print("[1/5] disponibilité")
    wait_until_ready(base_url)

    print("[2/5] carte d'identité du modèle")
    status, info = call(base_url, "/model/info")
    if status != 200:
        raise SmokeTestError(f"GET /model/info a répondu {status}")
    print(f"  {info['model_name']} v{info['model_version']}, seuil {info['decision_threshold']}, "
          f"{info['n_features']} features")

    print("[3/5] contrat de features")
    status, contrat = call(base_url, "/features")
    if status != 200:
        raise SmokeTestError(f"GET /features a répondu {status}")
    application = contrat["application_features"]
    print(f"  {len(application)} features de dossier, {len(contrat['history_features'])} d'historique")

    print("[4/5] prédiction sur un dossier valide")
    dossier = build_application_file(application)
    status, prediction = call(base_url, "/predict", {"features": dossier})
    if status != 200:
        raise SmokeTestError(f"POST /predict a répondu {status} : {prediction}")
    if not 0.0 <= prediction["probability"] <= 1.0:
        raise SmokeTestError(f"probabilité hors bornes : {prediction['probability']}")
    if prediction["decision"] not in {"accepted", "rejected"}:
        raise SmokeTestError(f"décision inattendue : {prediction['decision']}")
    attendu = "rejected" if prediction["probability"] >= prediction["threshold"] else "accepted"
    if prediction["decision"] != attendu:
        raise SmokeTestError("la décision ne suit pas le seuil annoncé")
    print(f"  probabilité {prediction['probability']:.4f} -> {prediction['decision']} "
          f"(seuil {prediction['threshold']})")

    print("[5/5] rejet d'une entrée invalide")
    status, erreur = call(base_url, "/predict", {"features": {"FEATURE_INEXISTANTE": 1.0}})
    if status != 422:
        raise SmokeTestError(f"une entrée invalide a répondu {status} au lieu de 422")
    print(f"  refus correct : {erreur['detail'][:70]}...")

    print("\nTest de fumée réussi : le service est opérationnel.")
    return 0


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    try:
        sys.exit(main(url))
    except SmokeTestError as error:
        print(f"\nECHEC : {error}", file=sys.stderr)
        sys.exit(1)
