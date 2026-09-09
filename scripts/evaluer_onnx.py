"""Évalue le passage du modèle LightGBM à ONNX Runtime : fidélité et vitesse.

La fidélité est mesurée sur les dossiers réels de la Partie 1 (hors dépôt), la
vitesse sur un dossier unitaire et sur un lot, à un thread de calcul comme en
production (OMP_NUM_THREADS=1).

    uv run --group perf python scripts/evaluer_onnx.py [--lignes 50000] [--sortie docs/perf/onnx.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import chronometrer, contexte_de_mesure

P6_DEFAUT = r"C:\Users\ClementLoire\Documents\OpenClassrooms\P6 - Initiez-vous au MLOps 1-2"


def convertir(booster, n_features: int):
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    # Le convertisseur LightGBM n'accepte que des entrées float32.
    return onnxmltools.convert_lightgbm(
        booster,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=15,
        zipmap=False,
    )


def session(modele_onnx):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        modele_onnx.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def dossiers_reels(noms: list[str], lignes: int) -> np.ndarray:
    import pandas as pd

    parquet = Path(os.environ.get("P6_PROJECT_ROOT", P6_DEFAUT)) / "output" / "feature_dataset.parquet"
    if not parquet.exists():
        raise SystemExit(f"Parquet de la Partie 1 introuvable : {parquet}")
    frame = pd.read_parquet(parquet)
    X = frame.drop(columns=["TARGET", "SK_ID_CURR"])
    X = X.drop(columns=X.select_dtypes(include="object").columns)
    X = X.replace([np.inf, -np.inf], np.nan)[noms].astype("float64")
    if lignes and lignes < len(X):
        X = X.sample(n=lignes, random_state=0)
    return X.to_numpy()


def main() -> int:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--lignes", type=int, default=50_000)
    parseur.add_argument("--sortie", type=Path)
    args = parseur.parse_args()

    from api.model import ScoringModel

    modele = ScoringModel.load()
    booster, seuil = modele._booster, modele.threshold
    onnx_modele = convertir(booster, len(modele.feature_names))
    sess = session(onnx_modele)

    def onnx_predict(X: np.ndarray) -> np.ndarray:
        return sess.run(["probabilities"], {"input": X.astype(np.float32)})[0][:, 1]

    X = dossiers_reels(modele.feature_names, args.lignes)
    reference = booster.predict(X)
    obtenu = np.concatenate([onnx_predict(X[i : i + 5000]) for i in range(0, len(X), 5000)])
    ecart = np.abs(obtenu - reference)
    divergentes = (reference >= seuil) != (obtenu >= seuil)

    fidelite = {
        "dossiers": len(X),
        "ecart_max": float(ecart.max()),
        "ecart_moyen": float(ecart.mean()),
        "ecart_p99": float(np.quantile(ecart, 0.99)),
        "dossiers_ecart_sup_1e-4": int((ecart > 1e-4).sum()),
        "decisions_divergentes": int(divergentes.sum()),
    }

    unitaire, lot = X[:1], X[:200]
    vitesse = {
        "lightgbm_unitaire": chronometrer(lambda: booster.predict(unitaire, num_threads=1), 3000),
        "onnx_unitaire": chronometrer(lambda: onnx_predict(unitaire), 3000),
        "lightgbm_lot_200": chronometrer(lambda: booster.predict(lot, num_threads=1), 200),
        "onnx_lot_200": chronometrer(lambda: onnx_predict(lot), 200),
    }

    # La quantification int8 d'ONNX Runtime porte sur les opérateurs à poids
    # (MatMul, Conv, Gemm...). Un ensemble d'arbres n'en contient aucun.
    operateurs = sorted({noeud.op_type for noeud in onnx_modele.graph.node})

    resultats = {
        "fidelite": fidelite,
        "vitesse_ms": vitesse,
        "operateurs_onnx": operateurs,
        "taille_onnx_mo": round(len(onnx_modele.SerializeToString()) / 1e6, 2),
        "taille_lightgbm_mo": round(
            (PROJECT_ROOT / "models" / "credit_default_lgbm.txt").stat().st_size / 1e6, 2
        ),
    }
    print(json.dumps(resultats, indent=2, ensure_ascii=False))

    if args.sortie:
        args.sortie.parent.mkdir(parents=True, exist_ok=True)
        args.sortie.write_text(
            json.dumps({**contexte_de_mesure(), **resultats}, indent=2, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
