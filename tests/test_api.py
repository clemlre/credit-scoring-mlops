"""Tests des routes HTTP : contrat, documentation, erreurs.

L'objectif n'est pas de retester le modèle (c'est le rôle de `test_model.py`) mais
de vérifier que l'API traduit correctement chaque situation en code de statut et en
message exploitable par l'appelant.
"""

from __future__ import annotations

import pytest

from api import config


class TestRoutesDeService:
    def test_le_service_est_disponible(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "1"

    def test_la_documentation_openapi_est_generee(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        chemins = schema["paths"]
        assert {"/health", "/model/info", "/features", "/predict", "/predict/batch"} <= set(chemins)

    def test_swagger_est_accessible(self, client):
        assert client.get("/docs").status_code == 200

    def test_chaque_route_documente_ses_reponses_derreur(self, client):
        schema = client.get("/openapi.json").json()
        for chemin, methode in [("/predict", "post"), ("/predict/batch", "post")]:
            reponses = schema["paths"][chemin][methode]["responses"]
            assert "422" in reponses, chemin
            assert "503" in reponses, chemin


class TestRoutesDeModele:
    def test_la_carte_didentite_du_modele_est_complete(self, client):
        body = client.get("/model/info").json()
        assert body["model_name"] == "credit-default-lgbm"
        assert body["decision_threshold"] == 0.1
        assert body["n_features"] == 779
        assert body["n_trees"] == 867
        assert body["metrics"]["auc_oof"] == pytest.approx(0.7888, abs=1e-4)
        # La traçabilité vers l'entraînement doit être exposée : sans elle, on ne peut
        # pas relier un score servi en production au run qui l'a produit.
        assert len(body["source_run_id"]) == 32

    def test_le_seuil_est_justifie_dans_la_reponse(self, client):
        """Un seuil de 0,10 sans explication est incompréhensible pour un métier."""
        body = client.get("/model/info").json()
        assert "10" in body["threshold_rationale"]

    def test_le_contrat_de_features_est_expose(self, client):
        body = client.get("/features").json()
        assert body["n_features"] == 779
        assert len(body["application_features"]) == 245
        assert len(body["history_features"]) == 534
        assert "AMT_CREDIT" in body["application_features"]
        assert all(f.startswith(config.HISTORY_PREFIXES) for f in body["history_features"])


class TestPredictionNominale:
    def test_un_dossier_complet_est_score(self, client, valid_features):
        response = client.post("/predict", json={"features": valid_features})
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["decision"] in {"accepted", "rejected"}
        assert body["threshold"] == 0.1
        assert body["model_version"] == "1"

    def test_la_reponse_dit_sur_quelle_information_elle_repose(self, client, valid_features):
        body = client.post("/predict", json={"features": valid_features}).json()
        couverture = body["coverage"]
        assert couverture["features_provided"] == 245
        assert couverture["features_missing"] == 534
        assert couverture["application_ratio"] == 1.0
        assert couverture["history_ratio"] == 0.0

    def test_la_decision_est_coherente_avec_le_seuil(self, client, valid_features):
        body = client.post("/predict", json={"features": valid_features}).json()
        attendu = "rejected" if body["probability"] >= body["threshold"] else "accepted"
        assert body["decision"] == attendu

    def test_un_dossier_risque_est_refuse(self, client, valid_features):
        risque = {
            **valid_features,
            **{n: 0.05 for n in valid_features if n.startswith("EXT_SOURCE")},
        }
        body = client.post("/predict", json={"features": risque}).json()
        assert body["decision"] == "rejected"

    def test_lhistorique_de_credit_est_facultatif(self, client, valid_features):
        """Un primo-emprunteur n'a pas d'historique : son dossier doit rester scorable."""
        response = client.post("/predict", json={"features": valid_features})
        assert response.status_code == 200
        assert response.json()["coverage"]["history_ratio"] == 0.0


class TestGestionDesErreurs:
    def test_un_nom_de_feature_inconnu_est_refuse(self, client, valid_features):
        """Ignorer silencieusement une faute de frappe donnerait un score calculé
        sans la variable que l'appelant croit avoir transmise."""
        payload = {**valid_features, "AMT_CREDITT": 100.0}
        response = client.post("/predict", json={"features": payload})
        assert response.status_code == 422
        assert "AMT_CREDITT" in response.json()["detail"]
        assert "GET /features" in response.json()["detail"]

    def test_la_liste_des_noms_inconnus_est_bornee(self, client, valid_features):
        """Un payload truffé de fautes ne doit pas produire une erreur de 100 ko."""
        payload = {**valid_features, **{f"INCONNUE_{i}": 1.0 for i in range(50)}}
        response = client.post("/predict", json={"features": payload})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "50 feature(s) inconnue(s)" in detail
        assert "et 40 autres" in detail

    @pytest.mark.parametrize(
        ("name", "value", "raison"),
        [
            ("EXT_SOURCE_2", 1.5, "score externe hors [0,1]"),
            ("AMT_CREDIT", -5000.0, "montant négatif"),
            ("CNT_CHILDREN", -1.0, "effectif négatif"),
            ("FLAG_OWN_CAR", 7.0, "indicateur non binaire"),
            ("DAYS_BIRTH", 9461.0, "jours positifs au lieu de négatifs"),
        ],
    )
    def test_une_valeur_hors_plage_est_refusee(self, client, valid_features, name, value, raison):
        payload = {**valid_features, name: value}
        response = client.post("/predict", json={"features": payload})
        assert response.status_code == 422, raison
        assert name in response.json()["detail"]

    def test_un_dossier_trop_incomplet_est_refuse(self, client, sparse_features):
        response = client.post("/predict", json={"features": sparse_features})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "trop incomplet" in detail
        # Le message doit dire quoi corriger, pas seulement que c'est refusé.
        assert "minimum requis" in detail

    def test_un_payload_vide_est_refuse(self, client):
        response = client.post("/predict", json={"features": {}})
        assert response.status_code == 422

    def test_un_corps_sans_champ_features_est_refuse(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "valeur",
        ["pas un nombre", "12.5", True, [1, 2], {"a": 1}],
        ids=["texte", "nombre en chaine", "booleen", "liste", "objet"],
    )
    def test_un_type_incorrect_est_refuse(self, client, valid_features, valeur):
        """`"12.5"` est rejeté aussi : accepter les chaînes obligerait à deviner si
        `"0,5"` vaut 0.5 ou 5, et une locale mal devinée fausse un revenu sans bruit."""
        payload = {**valid_features, "AMT_CREDIT": valeur}
        response = client.post("/predict", json={"features": payload})
        assert response.status_code == 422

    @pytest.mark.parametrize("litteral", ["Infinity", "-Infinity", "NaN"])
    def test_les_valeurs_non_finies_sont_refusees(self, client, valid_features, litteral):
        """`Infinity` et `NaN` ne font pas partie du JSON standard, mais le parseur de
        Python les accepte : ils franchiraient donc la couche transport sans bruit et
        fausseraient les comparaisons de seuil dans les arbres. On les envoie ici en
        contenu brut, car le client HTTP refuse lui-même de les sérialiser."""
        import json

        payload = {**valid_features, "AMT_CREDIT": 0.0}
        corps = json.dumps({"features": payload}).replace('"AMT_CREDIT": 0.0', f'"AMT_CREDIT": {litteral}')
        response = client.post(
            "/predict", content=corps, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_une_valeur_nulle_est_acceptee_comme_non_renseignee(self, client, valid_features):
        payload = {**valid_features, "AMT_GOODS_PRICE": None}
        response = client.post("/predict", json={"features": payload})
        assert response.status_code == 200
        assert response.json()["coverage"]["features_provided"] == 244

    def test_un_json_malforme_est_refuse(self, client):
        response = client.post(
            "/predict", content="{ceci n'est pas du json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestModeDegrade:
    @pytest.fixture
    def sans_modele(self, client):
        """Simule un artefact de modèle indisponible, puis restaure l'état."""
        from api.main import app

        original = app.state.model
        app.state.model = None
        app.state.model_error = "artefact absent (simulé par le test)"
        yield
        app.state.model = original

    def test_le_service_se_declare_indisponible(self, client, sans_modele):
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"
        assert response.json()["model_loaded"] is False

    def test_la_prediction_est_refusee_avec_la_cause(self, client, sans_modele, valid_features):
        response = client.post("/predict", json={"features": valid_features})
        assert response.status_code == 503
        assert "artefact absent" in response.json()["detail"]

    def test_les_routes_dinformation_sont_aussi_refusees(self, client, sans_modele):
        assert client.get("/model/info").status_code == 503
        assert client.get("/features").status_code == 503


class TestDemarrageSansArtefact:
    """Le vrai scénario de mode dégradé : le service démarre alors que l'artefact
    de modèle est introuvable (volume mal monté, image mal construite)."""

    def test_le_service_demarre_quand_meme_et_explique_ce_qui_manque(self, tmp_path, monkeypatch):
        import asyncio

        from fastapi import FastAPI

        from api import config as api_config
        from api.main import lifespan

        monkeypatch.setattr(api_config, "MODEL_DIR", tmp_path)

        async def demarrer():
            application = FastAPI()
            async with lifespan(application):
                # Pas de crash au démarrage : on peut diagnostiquer un conteneur
                # qui répond, pas un conteneur qui redémarre en boucle.
                assert application.state.model is None
                assert "incomplet" in application.state.model_error

        asyncio.run(demarrer())


class TestErreurInterne:
    def test_une_erreur_imprevue_ne_fuite_aucune_trace(self, client, valid_features, monkeypatch):
        """Une pile d'exécution exposerait les chemins du serveur et les versions
        de bibliothèques installées. Le client ne doit voir qu'un message neutre."""
        from fastapi.testclient import TestClient

        from api.main import app

        def exploser(_rows):
            raise RuntimeError("panne simulée dans le moteur d'inférence")

        monkeypatch.setattr(app.state.model, "predict", exploser)

        # Pas de `with` ici : ouvrir un second contexte relancerait le `lifespan` et,
        # en le refermant, viderait le modèle chargé pour toute la session de test.
        # L'application est déjà démarrée par la fixture `client`.
        client_brut = TestClient(app, raise_server_exceptions=False)
        response = client_brut.post("/predict", json={"features": valid_features})

        assert response.status_code == 500
        assert response.json() == {"detail": "Erreur interne du service."}
        assert "panne simulée" not in response.text


class TestPredictionParLot:
    def test_un_lot_est_score(self, client, valid_features):
        autre = {**valid_features, "EXT_SOURCE_2": 0.2}
        response = client.post(
            "/predict/batch",
            json={"items": [{"features": valid_features}, {"features": autre}]},
        )
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 2

    def test_le_lot_donne_le_meme_score_que_lappel_unitaire(self, client, valid_features):
        unitaire = client.post("/predict", json={"features": valid_features}).json()
        lot = client.post("/predict/batch", json={"items": [{"features": valid_features}]}).json()
        assert lot["predictions"][0]["probability"] == unitaire["probability"]

    def test_un_lot_vide_est_refuse(self, client):
        response = client.post("/predict/batch", json={"items": []})
        assert response.status_code == 422

    def test_un_lot_trop_grand_est_refuse(self, client, valid_features):
        """Sans plafond, un seul appel peut réclamer des millions de scores."""
        items = [{"features": valid_features}] * (config.MAX_BATCH_SIZE + 1)
        response = client.post("/predict/batch", json={"items": items})
        assert response.status_code == 422
        assert str(config.MAX_BATCH_SIZE) in response.json()["detail"]

    def test_lerreur_designe_la_demande_fautive(self, client, valid_features, sparse_features):
        """Dans un lot de 500, « une entrée est invalide » est inexploitable."""
        response = client.post(
            "/predict/batch",
            json={
                "items": [
                    {"features": valid_features},
                    {"features": valid_features},
                    {"features": sparse_features},
                ]
            },
        )
        assert response.status_code == 422
        assert "Demande n°2" in response.json()["detail"]


class TestJournalisationDesPredictions:
    """Ce que l'API promet côté monitoring, vérifié depuis l'extérieur.

    Ces tests ne touchent pas à PostgreSQL : ils vérifient le **contrat HTTP** et
    l'invariant « le monitoring ne casse jamais une prédiction ». La validité du
    SQL est couverte séparément, dans `test_storage.py`.
    """

    @pytest.fixture
    def journal_espion(self, client):
        """Remplace le journal par un espion, puis restaure l'original."""
        from api.main import app

        class Espion:
            def __init__(self):
                self.recus = []

            def record(self, records):
                self.recus.extend(records)

            def status(self):
                return {"stdout": True, "database": "ready", "last_error": None}

        original = app.state.prediction_log
        espion = Espion()
        app.state.prediction_log = espion
        yield espion
        app.state.prediction_log = original

    def test_chaque_reponse_porte_un_identifiant_de_requete(self, client, valid_features):
        """Sans identifiant renvoyé, impossible de relier une réclamation client à
        la ligne correspondante en base."""
        response = client.post("/predict", json={"features": valid_features})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"]

    def test_deux_requetes_ont_des_identifiants_distincts(self, client, valid_features):
        premier = client.post("/predict", json={"features": valid_features})
        second = client.post("/predict", json={"features": valid_features})
        assert premier.headers["X-Request-ID"] != second.headers["X-Request-ID"]

    def test_une_prediction_est_journalisee(self, client, valid_features, journal_espion):
        response = client.post("/predict", json={"features": valid_features})

        assert len(journal_espion.recus) == 1
        enregistrement = journal_espion.recus[0]
        assert enregistrement.request_id == response.headers["X-Request-ID"]
        assert enregistrement.endpoint == "/predict"
        assert enregistrement.probability == response.json()["probability"]
        assert enregistrement.decision == response.json()["decision"]
        assert enregistrement.model_version == "1"
        assert enregistrement.latency_ms > 0

    def test_le_payload_recu_est_conserve_tel_quel(self, client, valid_features, journal_espion):
        """C'est cette copie qui permettra de mesurer la dérive des données."""
        client.post("/predict", json={"features": valid_features})
        assert journal_espion.recus[0].features == valid_features

    def test_un_lot_journalise_chaque_demande(self, client, valid_features, journal_espion):
        client.post(
            "/predict/batch",
            json={"items": [{"features": valid_features}] * 3},
        )
        assert len(journal_espion.recus) == 3
        assert {r.endpoint for r in journal_espion.recus} == {"/predict/batch"}

    def test_une_requete_refusee_n_est_pas_journalisee(
        self, client, sparse_features, journal_espion
    ):
        """Seules les prédictions réellement rendues entrent au journal : une 422
        n'a produit aucun score à surveiller."""
        response = client.post("/predict", json={"features": sparse_features})
        assert response.status_code == 422
        assert journal_espion.recus == []

    def test_une_panne_du_journal_ne_casse_pas_la_prediction(self, client, valid_features):
        """L'invariant central. Le score doit être rendu même si la journalisation
        échoue : une panne de monitoring n'est pas une panne de production."""
        from api.main import app

        class JournalEnPanne:
            def record(self, records):
                raise RuntimeError("panne de journalisation simulée")

            def status(self):
                return {"stdout": True, "database": "unavailable", "last_error": "simulée"}

        original = app.state.prediction_log
        app.state.prediction_log = JournalEnPanne()
        try:
            response = client.post("/predict", json={"features": valid_features})
        finally:
            app.state.prediction_log = original

        assert response.status_code == 200
        assert "probability" in response.json()

    def test_letat_du_journal_est_expose_par_health(self, client):
        body = client.get("/health").json()
        assert body["prediction_log"]["stdout"] is True
        # Sans DATABASE_URL en test, le stockage est désactivé — et c'est normal.
        assert body["prediction_log"]["database"] in {"ready", "disabled", "unavailable"}

    def test_une_base_en_panne_ne_rend_pas_le_service_indisponible(self, client):
        """Retirer l'API du trafic parce que la base de monitoring est tombée
        transformerait une panne d'observabilité en panne de service."""
        from api.main import app

        class JournalDegrade:
            def status(self):
                return {"stdout": True, "database": "unavailable", "last_error": "simulée"}

        original = app.state.prediction_log
        app.state.prediction_log = JournalDegrade()
        try:
            response = client.get("/health")
        finally:
            app.state.prediction_log = original

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["prediction_log"]["database"] == "unavailable"


def test_racine_redirige_vers_la_documentation(client):
    """La racine ne doit pas répondre 404 : c'est la porte d'entrée du service déployé."""
    reponse = client.get("/", follow_redirects=False)
    assert reponse.status_code in (302, 307)
    assert reponse.headers["location"] == "/docs"


def test_racine_absente_du_contrat_publie(client):
    """La redirection est un confort de navigation, pas un point d'entrée métier."""
    schema = client.get("/openapi.json").json()
    assert "/" not in schema["paths"]
    assert set(schema["paths"]) == {
        "/health", "/model/info", "/features", "/predict", "/predict/batch",
    }
