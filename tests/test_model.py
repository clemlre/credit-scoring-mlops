"""Tests de la logique de scoring, sans HTTP.

Ce fichier vérifie le cœur : contrat de features, garde-fous, traitement des
valeurs manquantes et décision au seuil métier. Aucun serveur n'est lancé — si un
de ces tests casse, le problème est dans le modèle, pas dans l'API.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from api.model import ModelLoadError, ScoringModel

# Probabilité de référence pour le dossier synthétique complet. Figée volontairement :
# si l'artefact de modèle change sans qu'on l'ait décidé, ce test le signale
# immédiatement au lieu de laisser passer un modèle différent en production.
REFERENCE_PROBABILITY = 0.032803005390608875


class TestChargement:
    def test_le_contrat_du_modele_est_celui_attendu(self, model):
        assert len(model.feature_names) == 779
        assert model.threshold == 0.1
        assert model.version == "1"

    def test_les_features_sont_ventilees_entre_dossier_et_historique(self, model):
        assert len(model.application_features) == 245
        assert len(model.history_features) == 534
        # Les deux ensembles partitionnent le contrat : ni chevauchement, ni oubli.
        assert model.application_features & model.history_features == set()
        assert len(model.application_features | model.history_features) == 779

    def test_un_artefact_absent_donne_une_erreur_explicite(self, tmp_path):
        with pytest.raises(ModelLoadError, match="incomplet"):
            ScoringModel.load(tmp_path)

    def test_un_artefact_corrompu_donne_une_erreur_explicite(self, tmp_path):
        (tmp_path / "credit_default_lgbm.txt").write_text("ceci n'est pas un modele")
        (tmp_path / "feature_names.json").write_text("[]")
        (tmp_path / "model_metadata.json").write_text("{}")
        with pytest.raises(ModelLoadError):
            ScoringModel.load(tmp_path)

    def test_un_fichier_de_features_desaccorde_empeche_le_demarrage(self, tmp_path, model):
        """Le pire scénario : des colonnes décalées produiraient des scores faux mais
        plausibles. Le service doit refuser de démarrer plutôt que de servir ça."""
        from api import config

        (tmp_path / "credit_default_lgbm.txt").write_text(
            config.MODEL_FILE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        # On retire volontairement une feature de la liste.
        truncated = model.feature_names[:-1]
        (tmp_path / "feature_names.json").write_text(json.dumps(truncated), encoding="utf-8")
        (tmp_path / "model_metadata.json").write_text(
            config.METADATA_FILE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        with pytest.raises(ModelLoadError, match="Incohérence"):
            ScoringModel.load(tmp_path)


class TestContratDEntree:
    def test_les_noms_inconnus_sont_detectes(self, model, valid_features):
        payload = {**valid_features, "FEATURE_QUI_NEXISTE_PAS": 1.0, "AUTRE_INCONNUE": 2.0}
        assert model.unknown_features(payload) == ["AUTRE_INCONNUE", "FEATURE_QUI_NEXISTE_PAS"]

    def test_un_dossier_valide_na_aucun_nom_inconnu(self, model, valid_features):
        assert model.unknown_features(valid_features) == []

    def test_une_valeur_nulle_ne_compte_pas_comme_renseignee(self, model, valid_features):
        avec_trous = {**valid_features}
        for name in sorted(model.application_features)[:50]:
            avec_trous[name] = None

        pleine = model.coverage(valid_features)
        trouee = model.coverage(avec_trous)

        assert pleine.provided == 245
        assert trouee.provided == 195
        assert trouee.application_ratio < pleine.application_ratio

    def test_la_couverture_compte_separement_dossier_et_historique(self, model, valid_features):
        coverage = model.coverage(valid_features)
        assert coverage.application_ratio == 1.0
        assert coverage.history_ratio == 0.0
        assert coverage.missing == 779 - 245


class TestPlagesDeValidite:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("EXT_SOURCE_2", 1.5),
            ("EXT_SOURCE_2", -0.1),
            ("AMT_CREDIT", -1.0),
            ("CNT_CHILDREN", -3.0),
            ("FLAG_OWN_CAR", 2.0),
            ("DAYS_BIRTH", 100.0),
        ],
    )
    def test_les_valeurs_aberrantes_sont_signalees(self, model, valid_features, name, value):
        payload = {**valid_features, name: value}
        problems = model.out_of_range_features(payload)
        assert any(p.startswith(f"{name}=") for p in problems), problems

    def test_les_jours_negatifs_sont_normaux(self, model, valid_features):
        """Piège du jeu de données : `DAYS_BIRTH` vaut −9461 pour un client de 26 ans.
        Une validation « l'âge doit être positif » rejetterait tous les dossiers."""
        payload = {**valid_features, "DAYS_BIRTH": -9461.0, "DAYS_EMPLOYED": -637.0}
        assert model.out_of_range_features(payload) == []

    def test_les_ratios_derives_echappent_a_la_regle_des_jours(self, model, valid_features):
        """`DAYS_EMPLOYED_PERC` porte le préfixe `DAYS_` mais c'est un ratio positif."""
        payload = {**valid_features, "DAYS_EMPLOYED_PERC": 0.67}
        assert model.out_of_range_features(payload) == []

    def test_un_dossier_valide_ne_declenche_aucune_alerte(self, model, valid_features):
        assert model.out_of_range_features(valid_features) == []


class TestInference:
    def test_les_features_absentes_deviennent_des_nan(self, model):
        """Elles ne deviennent pas 0 : LightGBM sait router une valeur manquante,
        alors que 0 est une valeur *observée* qui déplacerait la prédiction."""
        matrix = model._to_matrix([{"AMT_CREDIT": 1000.0}])
        position = model.feature_names.index("AMT_CREDIT")
        assert matrix[0, position] == 1000.0
        assert np.isnan(matrix[0]).sum() == 778

    def test_la_prediction_de_reference_est_stable(self, model, valid_features):
        prediction = model.predict([valid_features])[0]
        assert prediction.probability == pytest.approx(REFERENCE_PROBABILITY, rel=1e-9)
        assert prediction.decision == "accepted"

    def test_la_decision_suit_le_seuil_metier(self, model, valid_features):
        risque = {
            **valid_features,
            **{name: 0.05 for name in valid_features if name.startswith("EXT_SOURCE")},
        }
        prediction = model.predict([risque])[0]
        assert prediction.probability >= model.threshold
        assert prediction.decision == "rejected"

    def test_un_score_externe_plus_faible_augmente_le_risque(self, model, valid_features):
        """Cohérence métier : le modèle doit réagir dans le bon sens."""
        probabilities = []
        for score in (0.5, 0.2, 0.05):
            payload = {
                **valid_features,
                **{n: score for n in valid_features if n.startswith("EXT_SOURCE")},
            }
            probabilities.append(model.predict([payload])[0].probability)
        assert probabilities == sorted(probabilities)

    def test_un_lot_vide_ne_provoque_pas_dappel_au_modele(self, model):
        assert model.predict([]) == []

    def test_le_lot_donne_le_meme_resultat_que_les_appels_unitaires(self, model, valid_features):
        autre = {**valid_features, "EXT_SOURCE_2": 0.2}
        lot = model.predict([valid_features, autre])
        unitaires = [model.predict([valid_features])[0], model.predict([autre])[0]]
        assert [p.probability for p in lot] == [p.probability for p in unitaires]


class TestFideliteAuModeleSource:
    def test_lapi_reproduit_exactement_le_modele_sur_des_clients_reels(self, model, real_clients):
        """Ignoré si les données de la Partie 1 ne sont pas montées."""
        import lightgbm as lgb

        from api import config

        booster = lgb.Booster(model_str=config.MODEL_FILE.read_text(encoding="utf-8"))
        matrix = model._to_matrix(real_clients)
        attendu = booster.predict(matrix)
        obtenu = [p.probability for p in model.predict(real_clients)]
        assert np.max(np.abs(np.array(obtenu) - attendu)) == 0.0

    def test_aucun_client_reel_nest_rejete_par_les_garde_fous(self, model, real_clients):
        from api import config

        for index, dossier in enumerate(real_clients):
            assert model.out_of_range_features(dossier) == [], f"client {index}"
            couverture = model.coverage(dossier)
            assert couverture.application_ratio >= config.MIN_APPLICATION_COVERAGE, f"client {index}"
