"""Métriques du modèle en service, out-of-fold, pour LightGBM et pour sa conversion ONNX.

Rejoue le protocole de la Partie 1 (src/optimize_lgbm.py : StratifiedKFold à 3 plis,
random_state=42, hyperparamètres Optuna, 867 arbres) sur les clients étiquetés du jeu
d'entraînement, puis calcule au seuil métier (0,10) et au seuil naïf (0,50) : AUC,
accuracy, précision, rappel, F1, coût métier, matrice de confusion et taux de refus.

Chaque modèle de pli est aussi converti en ONNX et évalué sur le même pli : les deux
moteurs sont comparés sur les mêmes dossiers, métrique par métrique.

Enfin, des dossiers réels passent par le chemin de l'API (ScoringModel.predict) et sont
comparés au booster brut : un écart nul prouve que les optimisations de code n'ont pas
touché au modèle.

    P6_PROJECT_ROOT=... uv run --group training --group perf python scripts/evaluer_modele.py \
        [--sortie docs/perf/metriques-modele.json] [--mettre-a-jour-metadata]

`--echantillon N` limite le calcul à N clients pour un essai rapide (le contrôle de
reproductibilité avec la Partie 1 n'a alors plus de sens et n'est pas fait).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import PROJECT_ROOT, contexte_de_mesure  # noqa: E402

PARAMETRES_PARTIE_1 = PROJECT_ROOT / "models" / "optuna_best_params.json"
METADATA = PROJECT_ROOT / "models" / "model_metadata.json"

# Coût métier de la Partie 1 : un mauvais client accepté coûte dix bons clients refusés.
COUT_FAUX_NEGATIF = 10
COUT_FAUX_POSITIF = 1
SEUIL_NAIF = 0.5

N_PLIS = 3
GRAINE = 42
TAILLE_LOT_ONNX = 5_000
TAILLE_LOT_API = 1_000


def auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Aire sous la courbe ROC par les rangs (Mann-Whitney), égalités comptées pour moitié."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(proba, dtype=float)
    ordre = np.argsort(p, kind="mergesort")
    _, inverse, effectifs = np.unique(p[ordre], return_inverse=True, return_counts=True)
    premiers = np.cumsum(effectifs) - effectifs + 1
    rangs = np.empty(len(p))
    rangs[ordre] = (premiers + (effectifs - 1) / 2)[inverse]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((rangs[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metriques_au_seuil(y_true: np.ndarray, proba: np.ndarray, seuil: float) -> dict:
    """Métriques de classification une fois la probabilité tranchée au seuil."""
    y = np.asarray(y_true).astype(bool)
    refus = np.asarray(proba, dtype=float) >= seuil
    tp = int((refus & y).sum())
    fp = int((refus & ~y).sum())
    fn = int((~refus & y).sum())
    tn = int((~refus & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    rappel = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
    return {
        "seuil": seuil,
        "auc": auc(y, proba),
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "rappel": rappel,
        "f1": f1,
        "cout_metier": COUT_FAUX_NEGATIF * fn + COUT_FAUX_POSITIF * fp,
        "taux_refus": (tp + fp) / len(y),
        "vrais_positifs": tp,
        "faux_positifs": fp,
        "faux_negatifs": fn,
        "vrais_negatifs": tn,
    }


def ecart_de_metriques(reference: dict, obtenu: dict) -> dict:
    """Différence obtenu − référence, métrique par métrique (le seuil n'en est pas une)."""
    return {nom: obtenu[nom] - reference[nom] for nom in reference if nom != "seuil"}


def comparer_probabilites(reference: np.ndarray, obtenu: np.ndarray, seuil: float) -> dict:
    """Écarts entre deux moteurs qui devraient rendre les mêmes probabilités."""
    ecart = np.abs(np.asarray(obtenu, dtype=float) - np.asarray(reference, dtype=float))
    divergentes = (reference >= seuil) != (obtenu >= seuil)
    return {
        "dossiers": int(len(ecart)),
        "ecart_max": float(ecart.max()),
        "ecart_moyen": float(ecart.mean()),
        "ecart_p99": float(np.quantile(ecart, 0.99)),
        "dossiers_ecart_sup_1e-4": int((ecart > 1e-4).sum()),
        "decisions_divergentes": int(divergentes.sum()),
    }


def parametres_du_modele(chemin: Path = PARAMETRES_PARTIE_1) -> dict:
    """Hyperparamètres Optuna de la Partie 1, avec le nombre d'arbres retenu."""
    document = json.loads(chemin.read_text(encoding="utf-8"))
    return {**document["best_params"], "n_estimators": int(document["best_iteration"])}


def reference_partie_1(chemin: Path = PARAMETRES_PARTIE_1) -> dict:
    document = json.loads(chemin.read_text(encoding="utf-8"))
    return {
        "auc_oof": float(document["auc_oof"]),
        "business_cost_optimal_threshold": int(document["business_cost_optimal_threshold"]),
    }


def controle_reproductibilite(
    auc_obtenue: float,
    cout_obtenu: int,
    reference: dict,
    tolerance_auc: float = 1e-3,
    tolerance_cout: float = 0.01,
) -> list[str]:
    """Avertissements si le rejeu ne retrouve pas les chiffres publiés par la Partie 1."""
    avertissements = []
    auc_attendue = reference["auc_oof"]
    if abs(auc_obtenue - auc_attendue) > tolerance_auc:
        avertissements.append(
            f"AUC OOF {auc_obtenue:.4f} au lieu de {auc_attendue:.4f} dans la Partie 1"
        )
    cout_attendu = reference["business_cost_optimal_threshold"]
    if abs(cout_obtenu - cout_attendu) / cout_attendu > tolerance_cout:
        avertissements.append(
            f"coût métier {cout_obtenu} au lieu de {cout_attendu} dans la Partie 1"
        )
    return avertissements


def charger_donnees(p6_root: Path, noms: list[str], echantillon: int | None):
    """Clients étiquetés de la Partie 1, préparés comme dans src/training.load_training_data."""
    import pandas as pd

    parquet = p6_root / "output" / "feature_dataset.parquet"
    if not parquet.exists():
        raise SystemExit(f"Parquet de la Partie 1 introuvable : {parquet}")
    frame = pd.read_parquet(parquet)
    frame = frame[frame["TARGET"].notna()]
    if echantillon and echantillon < len(frame):
        frame = frame.sample(n=echantillon, random_state=GRAINE)
    y = frame["TARGET"].astype(int).to_numpy()
    X = frame[noms].astype("float32").replace([np.inf, -np.inf], np.nan)
    return X.reset_index(drop=True), y


def predictions_oof(X, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Probabilités out-of-fold de LightGBM et de sa conversion ONNX, pli par pli."""
    from evaluer_onnx import convertir, session
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold

    plis = StratifiedKFold(n_splits=N_PLIS, shuffle=True, random_state=GRAINE)
    oof_lgbm = np.zeros(len(y))
    oof_onnx = np.zeros(len(y))
    journal = []
    for numero, (train, valid) in enumerate(plis.split(X, y), start=1):
        debut = time.perf_counter()
        modele = LGBMClassifier(random_state=GRAINE, n_jobs=-1, verbose=-1, **params)
        modele.fit(X.iloc[train], y[train])
        duree_fit = time.perf_counter() - debut
        oof_lgbm[valid] = modele.predict_proba(X.iloc[valid])[:, 1]

        debut = time.perf_counter()
        sess = session(convertir(modele.booster_, X.shape[1]))
        duree_conversion = time.perf_counter() - debut
        X_valid = X.iloc[valid].to_numpy(dtype=np.float32)
        oof_onnx[valid] = np.concatenate(
            [
                sess.run(["probabilities"], {"input": X_valid[i : i + TAILLE_LOT_ONNX]})[0][:, 1]
                for i in range(0, len(X_valid), TAILLE_LOT_ONNX)
            ]
        )
        journal.append(
            {
                "pli": numero,
                "clients_valides": int(len(valid)),
                "auc_lightgbm": auc(y[valid], oof_lgbm[valid]),
                "auc_onnx": auc(y[valid], oof_onnx[valid]),
                "entrainement_s": round(duree_fit, 1),
                "conversion_onnx_s": round(duree_conversion, 1),
            }
        )
        print(
            f"  pli {numero}/{N_PLIS} : AUC LightGBM {journal[-1]['auc_lightgbm']:.4f}, "
            f"ONNX {journal[-1]['auc_onnx']:.4f} ({duree_fit:.0f} s)",
            flush=True,
        )
    return oof_lgbm, oof_onnx, journal


def verifier_chemin_api(X, lignes: int) -> dict:
    """Les probabilités rendues par ScoringModel.predict contre celles du booster brut."""
    from api.model import ScoringModel

    modele = ScoringModel.load()
    matrice = X.head(lignes).to_numpy(dtype=np.float64)
    reference = modele._booster.predict(matrice)
    noms = list(X.columns)
    obtenu = []
    for debut in range(0, len(matrice), TAILLE_LOT_API):
        dossiers = [
            {nom: float(v) for nom, v in zip(noms, ligne, strict=True) if not np.isnan(v)}
            for ligne in matrice[debut : debut + TAILLE_LOT_API]
        ]
        obtenu.extend(p.probability for p in modele.predict(dossiers))
    return comparer_probabilites(reference, np.array(obtenu), modele.threshold)


def mettre_a_jour_metadata(metier: dict, naif: dict, chemin: Path = METADATA) -> None:
    """Ajoute les métriques OOF à la carte d'identité servie par /model/info.

    La même exécution fournit l'AUC, les coûts métier et les métriques de classification :
    ils doivent être mis à jour ensemble pour éviter de servir un mélange de rejeux.
    Au seuil naïf 0,5, l'accuracy et le rappel suffisent à montrer pourquoi l'accuracy
    seule ne dit rien sur ce problème.
    """
    document = json.loads(chemin.read_text(encoding="utf-8"))
    document["metrics"].update(
        {
            "auc_oof": metier["auc"],
            "business_cost_optimal_threshold": metier["cout_metier"],
            "business_cost_threshold_0.5": naif["cout_metier"],
            "accuracy_oof": metier["accuracy"],
            "precision_oof": metier["precision"],
            "recall_oof": metier["rappel"],
            "f1_oof": metier["f1"],
            "rejection_rate_oof": metier["taux_refus"],
            "accuracy_oof_threshold_0.5": naif["accuracy"],
            "recall_oof_threshold_0.5": naif["rappel"],
        }
    )
    chemin.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def afficher_tableau(resultats: dict) -> None:
    colonnes = ("auc", "accuracy", "precision", "rappel", "f1", "cout_metier", "taux_refus")
    print(f"\n{'':22}" + "".join(f"{c:>12}" for c in colonnes))
    for seuil, moteurs in resultats.items():
        for moteur, m in moteurs.items():
            valeurs = "".join(
                f"{m[c]:>12}" if isinstance(m[c], int) else f"{m[c]:>12.4f}" for c in colonnes
            )
            print(f"{moteur + ' @ ' + seuil:22}{valeurs}")


def main() -> int:
    parseur = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parseur.add_argument(
        "--p6-root", type=Path, default=os.environ.get("P6_PROJECT_ROOT"),
        help="racine du projet de la Partie 1 (sinon P6_PROJECT_ROOT)",
    )
    parseur.add_argument("--echantillon", type=int, help="nombre de clients (essai rapide)")
    parseur.add_argument("--lignes-api", type=int, default=20_000)
    parseur.add_argument("--sortie", type=Path)
    parseur.add_argument("--mettre-a-jour-metadata", action="store_true")
    args = parseur.parse_args()
    if args.p6_root is None:
        parseur.error("indiquer --p6-root ou définir P6_PROJECT_ROOT")

    from api.model import ScoringModel

    modele = ScoringModel.load()
    seuil_metier = modele.threshold
    params = parametres_du_modele()

    print(f"Chargement des clients étiquetés de la Partie 1 ({args.p6_root})...", flush=True)
    X, y = charger_donnees(args.p6_root, modele.feature_names, args.echantillon)
    print(f"  {len(y)} clients, {X.shape[1]} features, {y.mean():.2%} de défauts", flush=True)

    print(f"Validation croisée à {N_PLIS} plis, {params['n_estimators']} arbres...", flush=True)
    oof_lgbm, oof_onnx, plis = predictions_oof(X, y, params)

    seuils = {"seuil_metier": seuil_metier, "seuil_naif": SEUIL_NAIF}
    resultats = {
        nom: {
            "lightgbm": metriques_au_seuil(y, oof_lgbm, seuil),
            "onnx": metriques_au_seuil(y, oof_onnx, seuil),
        }
        for nom, seuil in seuils.items()
    }
    afficher_tableau(resultats)
    for moteurs in resultats.values():
        moteurs["ecart_onnx"] = ecart_de_metriques(moteurs["lightgbm"], moteurs["onnx"])

    fidelite_onnx = comparer_probabilites(oof_lgbm, oof_onnx, seuil_metier)
    print(
        f"\nONNX contre LightGBM : écart max {fidelite_onnx['ecart_max']:.1e}, "
        f"{fidelite_onnx['decisions_divergentes']} décision(s) changée(s)"
    )

    avertissements = []
    if not args.echantillon:
        metier = resultats["seuil_metier"]["lightgbm"]
        avertissements = controle_reproductibilite(
            metier["auc"], metier["cout_metier"], reference_partie_1()
        )
        for texte in avertissements:
            print(f"! {texte}")
        if not avertissements:
            print("Reproductibilité : les chiffres de la Partie 1 sont retrouvés.")

    print(f"\nChemin de l'API sur {args.lignes_api} clients...", flush=True)
    chemin_api = verifier_chemin_api(X, args.lignes_api)
    print(
        f"  écart max {chemin_api['ecart_max']:.1e}, "
        f"{chemin_api['decisions_divergentes']} décision(s) changée(s)"
    )

    document = {
        **contexte_de_mesure(),
        "protocole": {
            "plis": N_PLIS,
            "graine": GRAINE,
            "clients": int(len(y)),
            "taux_de_defaut": float(y.mean()),
            "echantillon": bool(args.echantillon),
            "hyperparametres": params,
        },
        "metriques_oof": resultats,
        "fidelite_onnx": fidelite_onnx,
        "reproductibilite_partie_1": {
            "controle_effectue": not args.echantillon,
            "avertissements": avertissements,
        },
        "chemin_api": chemin_api,
        "plis": plis,
    }
    if args.sortie:
        args.sortie.parent.mkdir(parents=True, exist_ok=True)
        args.sortie.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\n-> {args.sortie}")
    if args.mettre_a_jour_metadata:
        if args.echantillon:
            raise SystemExit(
                "Refus : des métriques sur un échantillon n'ont rien à faire dans les métadonnées."
            )
        mettre_a_jour_metadata(
            resultats["seuil_metier"]["lightgbm"], resultats["seuil_naif"]["lightgbm"]
        )
        print(f"-> {METADATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
