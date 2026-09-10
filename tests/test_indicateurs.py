"""Tests des calculs du tableau de bord (PSI, fenêtres, alertes)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from monitoring import indicateurs as ind

PROFIL = Path(__file__).resolve().parents[1] / "monitoring" / "profil_reference.json"


@pytest.fixture
def alea():
    return np.random.default_rng(0)


@pytest.fixture
def profil(alea):
    reference = {"A": alea.normal(0, 1, 20_000), "B": alea.uniform(0, 1, 20_000)}
    return {
        "features": {
            nom: {**ind.profil_feature(valeurs), "importance": 1}
            for nom, valeurs in reference.items()
        }
    }


def requetes(appels=100, erreurs_4xx=0, erreurs_5xx=0, latence_p95_ms=5.0):
    return {
        "appels": appels,
        "erreurs_4xx": erreurs_4xx,
        "erreurs_5xx": erreurs_5xx,
        "latence_p95_ms": latence_p95_ms,
    }


class TestFormats:
    def test_les_nombres_sont_ecrits_a_la_francaise(self):
        assert ind.pourcentage(0.207) == "20,7 %"
        assert ind.decimal(0.25) == "0,25"
        assert ind.decimal(7.456, 1) == "7,5"


class TestFenetres:
    @pytest.mark.parametrize(
        ("fenetre", "attendu"),
        [
            (timedelta(hours=1), "minute"),
            (timedelta(days=1), "hour"),
            (timedelta(days=7), "day"),
            (None, "day"),
        ],
    )
    def test_le_pas_suit_la_longueur_de_la_fenetre(self, fenetre, attendu):
        assert ind.pas_temporel(fenetre) == attendu


class TestPeriodePrecise:
    MAINTENANT = datetime(2026, 9, 10, 18, 0, tzinfo=UTC)

    def test_une_heure_sans_fuseau_est_lue_en_utc(self):
        debut, fin = ind.periode_precise("2026-09-10T15:50", "2026-09-10T16:05", self.MAINTENANT)
        assert debut == datetime(2026, 9, 10, 15, 50, tzinfo=UTC)
        assert fin - debut == timedelta(minutes=15)

    def test_un_decalage_horaire_est_respecte(self):
        debut, _ = ind.periode_precise("2026-09-10T17:50+02:00", None, self.MAINTENANT)
        assert debut == datetime(2026, 9, 10, 15, 50, tzinfo=UTC)

    def test_sans_fin_la_periode_court_jusqu_a_maintenant(self):
        _, fin = ind.periode_precise("2026-09-10T15:50Z", None, self.MAINTENANT)
        assert fin == self.MAINTENANT

    @pytest.mark.parametrize(
        ("du", "au"), [("hier", None), ("2026-09-10T16:05", "2026-09-10T15:50")]
    )
    def test_une_periode_invalide_est_refusee(self, du, au):
        with pytest.raises(ValueError):
            ind.periode_precise(du, au, self.MAINTENANT)


class TestRepartition:
    def test_une_variable_continue_donne_neuf_bornes(self, alea):
        assert len(ind.bornes_deciles(alea.normal(size=5000))) == 9

    def test_une_variable_binaire_donne_des_bornes_dedoublonnees(self, alea):
        bornes = ind.bornes_deciles(alea.integers(0, 2, 5000).astype(float))
        assert bornes == sorted(set(bornes))
        assert len(bornes) <= 2

    def test_les_manquants_sont_exclus(self):
        valeurs = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
        assert ind.repartition(valeurs, [2.0]).tolist() == pytest.approx([1 / 3, 2 / 3])

    def test_une_serie_vide_donne_une_repartition_nulle(self):
        assert ind.repartition(np.array([]), [0.5]).tolist() == [0.0, 0.0]


class TestPsi:
    def test_deux_repartitions_identiques_donnent_zero(self):
        assert ind.psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0

    def test_une_classe_vide_ne_fait_pas_diverger_le_calcul(self):
        assert np.isfinite(ind.psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))

    @pytest.mark.parametrize(
        ("valeur", "attendu"),
        [(0.05, "stable"), (0.10, "modérée"), (0.2, "modérée"), (0.25, "significative")],
    )
    def test_les_bandes_de_lecture(self, valeur, attendu):
        assert ind.bande(valeur) == attendu


class TestDerive:
    def test_un_echantillon_de_la_meme_loi_reste_stable(self, profil, alea):
        production = {"A": alea.normal(0, 1, 3000), "B": alea.uniform(0, 1, 3000)}
        assert {d["bande"] for d in ind.derive(profil, production)} == {"stable"}

    def test_un_decalage_est_detecte_sur_la_bonne_feature(self, profil, alea):
        production = {"A": alea.normal(1, 1, 3000), "B": alea.uniform(0, 1, 3000)}
        derives = ind.derive(profil, production)
        assert derives[0]["feature"] == "A"
        assert derives[0]["bande"] == "significative"
        assert derives[1]["bande"] == "stable"

    def test_le_taux_de_manquants_est_mesure(self, profil, alea):
        valeurs = alea.normal(0, 1, 1000)
        valeurs[:250] = np.nan
        derives = {d["feature"]: d for d in ind.derive(profil, {"A": valeurs})}
        assert derives["A"]["manquant_production"] == 0.25
        assert derives["B"]["manquant_production"] is None


class TestAlertes:
    def test_un_service_sain_ne_declenche_rien(self):
        derives = [{"feature": "A", "bande": "stable"}]
        assert ind.alertes(requetes(), {"nombre": 1000}, derives) == []

    def test_une_erreur_interne_est_signalee(self):
        niveaux = [n for n, _ in ind.alertes(requetes(erreurs_5xx=1), {"nombre": 1000}, [])]
        assert niveaux == ["error"]

    def test_un_taux_de_refus_http_eleve_est_signale(self):
        messages = ind.alertes(requetes(erreurs_4xx=10), {"nombre": 1000}, [])
        assert messages[0][0] == "warning"
        assert "10,0 %" in messages[0][1]

    def test_une_latence_hors_objectif_est_signalee(self):
        messages = ind.alertes(requetes(latence_p95_ms=250.0), {"nombre": 1000}, [])
        assert "250 ms" in messages[0][1]

    def test_une_derive_significative_est_signalee(self):
        derives = [{"feature": "EXT_SOURCE_2", "bande": "significative"}]
        messages = ind.alertes(requetes(), {"nombre": 1000}, derives)
        assert "EXT_SOURCE_2" in messages[0][1]

    def test_sous_le_volume_minimal_le_psi_n_est_pas_interprete(self):
        derives = [{"feature": "EXT_SOURCE_2", "bande": "significative"}]
        messages = ind.alertes(requetes(), {"nombre": 100}, derives)
        assert [n for n, _ in messages] == ["info"]


def test_le_profil_versionne_est_coherent():
    profil = json.loads(PROFIL.read_text(encoding="utf-8"))
    assert len(profil["features"]) == 20
    for nom, feature in profil["features"].items():
        assert feature["bornes"] == sorted(feature["bornes"]), nom
        assert len(feature["proportions"]) == len(feature["bornes"]) + 1, nom
        assert sum(feature["proportions"]) == pytest.approx(1.0, abs=1e-4), nom
