"""Construit le profil de référence utilisé par le tableau de bord pour mesurer la dérive.

Même référence que le notebook de drift : 20 000 dossiers étiquetés tirés du jeu
d'entraînement de la Partie 1 (graine 42), et les 20 features au plus fort gain dans
le modèle servi. Le profil ne contient que des déciles et des proportions, aucune
ligne client : il peut être versionné.

    P6_PROJECT_ROOT=... uv run --group monitoring python -m monitoring.construire_reference
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.indicateurs import profil_feature

SORTIE = Path(__file__).with_name("profil_reference.json")
TAILLE_REFERENCE = 20_000
GRAINE = 42
NB_FEATURES = 20


def main() -> int:
    from api.model import ScoringModel

    modele = ScoringModel.load()
    gains = pd.Series(
        modele._booster.feature_importance(importance_type="gain"), index=modele.feature_names
    ).sort_values(ascending=False)
    suivies = gains.head(NB_FEATURES)

    racine = os.environ.get("P6_PROJECT_ROOT")
    if not racine:
        raise SystemExit("Définir P6_PROJECT_ROOT (racine du projet de la Partie 1).")
    parquet = Path(racine) / "output" / "feature_dataset.parquet"
    if not parquet.exists():
        raise SystemExit(f"Parquet de la Partie 1 introuvable : {parquet}")
    brut = pd.read_parquet(parquet)
    reference = (
        brut[brut["TARGET"].notna()]
        .sample(n=TAILLE_REFERENCE, random_state=GRAINE)
        .replace([np.inf, -np.inf], np.nan)
        .reindex(columns=suivies.index)
        .astype("float64")
    )

    profil = {
        "source": "feature_dataset.parquet de la Partie 1, dossiers étiquetés",
        "lignes": TAILLE_REFERENCE,
        "graine": GRAINE,
        "part_du_gain": round(float(suivies.sum() / gains.sum()), 4),
        "features": {
            nom: {**profil_feature(reference[nom].to_numpy()), "importance": int(gain)}
            for nom, gain in suivies.items()
        },
    }
    SORTIE.write_text(
        json.dumps(profil, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    print(f"{len(profil['features'])} features, {profil['part_du_gain']:.1%} du gain -> {SORTIE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
