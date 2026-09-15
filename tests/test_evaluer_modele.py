"""Tests des calculs de scripts/evaluer_modele.py : métriques à un seuil, comparaison de
deux moteurs d'inférence, contrôle de reproductibilité avec la Partie 1."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import evaluer_modele as em

# 3 défauts puis 7 bons clients : chaque chiffre attendu se recalcule à la main.
Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
PROBA = np.array([0.9, 0.2, 0.05, 0.3, 0.15, 0.08, 0.02, 0.01, 0.5, 0.04])


class TestMetriquesAuSeuil:
    def test_la_matrice_de_confusion_au_seuil_metier(self):
        m = em.metriques_au_seuil(Y, PROBA, 0.10)
        matrice = (m["vrais_positifs"], m["faux_positifs"], m["faux_negatifs"], m["vrais_negatifs"])
        assert matrice == (2, 3, 1, 4)

    def test_les_taux_derivent_de_la_matrice(self):
        m = em.metriques_au_seuil(Y, PROBA, 0.10)
        assert m["accuracy"] == pytest.approx(0.6)
        assert m["precision"] == pytest.approx(0.4)
        assert m["rappel"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(0.5)
        assert m["taux_refus"] == pytest.approx(0.5)

    def test_le_cout_metier_pese_dix_fois_un_faux_negatif(self):
        assert em.metriques_au_seuil(Y, PROBA, 0.10)["cout_metier"] == 13
        assert em.metriques_au_seuil(Y, PROBA, 0.50)["cout_metier"] == 21

    def test_lauc_ne_depend_pas_du_seuil(self):
        # 21 paires (défaut, bon client) : 15 sont ordonnées dans le bon sens.
        assert em.metriques_au_seuil(Y, PROBA, 0.10)["auc"] == pytest.approx(15 / 21)
        assert em.metriques_au_seuil(Y, PROBA, 0.50)["auc"] == pytest.approx(15 / 21)

    def test_les_egalites_de_score_comptent_pour_moitie(self):
        assert em.auc(np.array([1, 0]), np.array([0.3, 0.3])) == pytest.approx(0.5)

    def test_sans_aucun_refus_les_taux_valent_zero_sans_division_par_zero(self):
        m = em.metriques_au_seuil(Y, PROBA, 0.99)
        assert (m["precision"], m["rappel"], m["f1"], m["taux_refus"]) == (0.0, 0.0, 0.0, 0.0)


class TestComparaisonDeMoteurs:
    def test_les_ecarts_de_probabilite_et_les_decisions_changees(self):
        reference = np.array([0.10, 0.50, 0.09])
        obtenu = np.array([0.10, 0.5001, 0.11])
        c = em.comparer_probabilites(reference, obtenu, 0.10)
        assert c["ecart_max"] == pytest.approx(0.02)
        assert c["dossiers_ecart_sup_1e-4"] == 1
        assert c["decisions_divergentes"] == 1

    def test_deux_moteurs_identiques_ont_un_ecart_de_metriques_nul(self):
        m = em.metriques_au_seuil(Y, PROBA, 0.10)
        ecart = em.ecart_de_metriques(m, m)
        assert "seuil" not in ecart
        assert set(ecart) == set(m) - {"seuil"}
        assert all(valeur == 0 for valeur in ecart.values())


class TestParametresDuModele:
    def test_les_hyperparametres_sont_ceux_de_la_partie_1(self):
        params = em.parametres_du_modele()
        assert params["n_estimators"] == 867
        assert params["num_leaves"] == 96
        assert "threshold" not in params


class TestMiseAJourDesMetadonnees:
    def test_les_metriques_oof_sont_ajoutees_sans_toucher_au_reste(self, tmp_path):
        chemin = tmp_path / "model_metadata.json"
        chemin.write_text(
            json.dumps({"model_version": "1", "metrics": {"auc_oof": 0.7889}}), encoding="utf-8"
        )
        metier = em.metriques_au_seuil(Y, PROBA, 0.10)
        naif = em.metriques_au_seuil(Y, PROBA, 0.50)
        em.mettre_a_jour_metadata(metier, naif, chemin)
        document = json.loads(chemin.read_text(encoding="utf-8"))
        assert document["model_version"] == "1"
        assert document["metrics"]["auc_oof"] == 0.7889
        assert document["metrics"]["recall_oof"] == pytest.approx(2 / 3)
        assert document["metrics"]["rejection_rate_oof"] == pytest.approx(0.5)
        assert document["metrics"]["recall_oof_threshold_0.5"] == pytest.approx(1 / 3)
        assert document["metrics"]["accuracy_oof_threshold_0.5"] == pytest.approx(0.7)


class TestReproductibilite:
    REFERENCE = {"auc_oof": 0.78887, "business_cost_optimal_threshold": 150_877}

    def test_rien_a_signaler_quand_on_retrouve_la_partie_1(self):
        assert em.controle_reproductibilite(0.7889, 150_900, self.REFERENCE) == []

    def test_un_ecart_dauc_est_signale(self):
        avertissements = em.controle_reproductibilite(0.75, 150_877, self.REFERENCE)
        assert len(avertissements) == 1
        assert "AUC" in avertissements[0]

    def test_un_ecart_de_cout_est_signale(self):
        avertissements = em.controle_reproductibilite(0.7889, 160_000, self.REFERENCE)
        assert len(avertissements) == 1
        assert "coût" in avertissements[0]
