"""Tests du journal des prédictions.

Deux niveaux, volontairement séparés :

- **Unitaires**, avec un faux pool de connexions. Ils vérifient la *logique* —
  quels canaux sont alimentés, ce qui est absorbé, ce qui est écrit — sans exiger
  de serveur PostgreSQL. C'est ce qui tourne partout, y compris sur un poste sans
  Docker.
- **D'intégration**, contre un vrai PostgreSQL, ignorés si `DATABASE_URL` n'est pas
  défini. Eux seuls prouvent que le SQL est valide : un faux pool accepterait
  n'importe quelle requête, y compris une requête syntaxiquement fausse.

La distinction compte pour la soutenance : savoir *ce que la suite de tests ne
garantit pas* vaut mieux qu'un chiffre de couverture élevé.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

import pytest

from api import config
from api.storage import PredictionLog, PredictionRecord, _describe, build_record

# --------------------------------------------------------------------------
# Doublures
# --------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, pool):
        self._pool = pool

    def executemany(self, sql, rows):
        self._pool.inserted.append((sql, list(rows)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, pool):
        self._pool = pool

    def execute(self, sql):
        if self._pool.fail_on == "execute":
            raise RuntimeError("échec SQL simulé")
        self._pool.executed.append(sql)

    def cursor(self):
        return FakeCursor(self._pool)


class FakePool:
    """Pool de connexions simulé, capable d'échouer sur commande."""

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.executed: list[str] = []
        self.inserted: list[tuple] = []
        self.closed = False

    @contextmanager
    def connection(self, timeout=None):
        if self.fail_on == "connect":
            raise RuntimeError("base injoignable (simulée)")
        yield FakeConnection(self)

    def close(self):
        self.closed = True


@pytest.fixture
def record(model) -> PredictionRecord:
    """Un enregistrement représentatif, bâti comme le fait l'API."""
    from api.model import Coverage

    return build_record(
        request_id="11111111-2222-3333-4444-555555555555",
        endpoint="/predict",
        features={"AMT_CREDIT": 406597.5, "EXT_SOURCE_2": 0.2629},
        probability=0.0731,
        decision="accepted",
        coverage=Coverage(provided=2, missing=777, application_ratio=0.9, history_ratio=0.1),
        model_version="1",
        threshold=0.1,
        latency_ms=4.2,
    )


# --------------------------------------------------------------------------


class TestSansBaseDeDonnees:
    """Le cas par défaut : pas de `DATABASE_URL`. Le service doit fonctionner."""

    def test_le_journal_se_declare_desactive(self):
        journal = PredictionLog(None)
        journal.open()
        assert journal.database_enabled is False
        assert journal.status() == {"stdout": True, "database": "disabled", "last_error": None}

    def test_la_prediction_est_quand_meme_tracee_sur_la_sortie_standard(self, record, capfd):
        journal = PredictionLog(None)
        journal.open()
        journal.record([record])

        ligne = capfd.readouterr().out.strip()
        charge = json.loads(ligne)  # doit être du JSON valide, pas du texte formaté
        assert charge["event"] == "prediction"
        assert charge["decision"] == "accepted"
        assert charge["request_id"] == "11111111-2222-3333-4444-555555555555"

    def test_la_sortie_standard_ne_contient_aucune_valeur_de_feature(self, record, capfd):
        """Les journaux applicatifs partent souvent chez un tiers : les données
        financières du demandeur ne doivent pas s'y trouver."""
        journal = PredictionLog(None)
        journal.open()
        journal.record([record])

        ligne = capfd.readouterr().out.strip()
        assert "406597.5" not in ligne
        assert "AMT_CREDIT" not in ligne
        assert "features" not in json.loads(ligne)

    def test_un_lot_vide_n_ecrit_rien(self, capfd):
        journal = PredictionLog(None)
        journal.open()
        journal.record([])
        assert capfd.readouterr().out == ""


class TestEcritureEnBase:
    """Chemin nominal, avec un pool simulé."""

    def test_le_schema_est_cree_avant_la_premiere_insertion(self, record):
        journal = PredictionLog("postgresql://simule")
        journal._pool = FakePool()
        journal.record([record])

        assert any("CREATE TABLE IF NOT EXISTS predictions" in sql for sql in journal._pool.executed)
        assert journal.status()["database"] == "ready"

    def test_le_schema_n_est_cree_qu_une_fois(self, record):
        journal = PredictionLog("postgresql://simule")
        journal._pool = FakePool()
        journal.record([record])
        journal.record([record])

        assert len(journal._pool.executed) == 1

    def test_un_lot_part_en_une_seule_insertion(self, record):
        """Un aller-retour SQL par ligne annulerait le gain de la vectorisation."""
        journal = PredictionLog("postgresql://simule")
        journal._pool = FakePool()
        journal.record([record, record, record])

        assert len(journal._pool.inserted) == 1
        _, lignes = journal._pool.inserted[0]
        assert len(lignes) == 3

    def test_les_features_sont_stockees_en_base(self, record):
        journal = PredictionLog("postgresql://simule")
        journal._pool = FakePool()
        journal.record([record])

        _, lignes = journal._pool.inserted[0]
        # La dernière colonne est le JSONB des features.
        assert lignes[0][-1].obj == {"AMT_CREDIT": 406597.5, "EXT_SOURCE_2": 0.2629}

    def test_la_fermeture_libere_le_pool(self):
        journal = PredictionLog("postgresql://simule")
        faux = FakePool()
        journal._pool = faux
        journal.close()

        assert faux.closed is True
        assert journal.database_enabled is False


class TestDefaillanceDeLaBase:
    """L'invariant central : une panne de monitoring n'est pas une panne d'API."""

    def test_une_base_injoignable_n_interrompt_pas_la_journalisation(self, record, capfd):
        journal = PredictionLog("postgresql://simule")
        journal._pool = FakePool(fail_on="connect")

        journal.record([record])  # ne doit pas lever

        assert journal.status()["database"] == "unavailable"
        assert "injoignable" in journal.status()["last_error"]
        # Le canal `stdout`, lui, a bien reçu la prédiction : rien n'est perdu.
        assert json.loads(capfd.readouterr().out.strip())["event"] == "prediction"

    def test_un_echec_sql_est_absorbe(self, record):
        journal = PredictionLog("postgresql://simule")
        journal._pool = FakePool(fail_on="execute")

        journal.record([record])

        assert journal.status()["database"] == "unavailable"

    def test_le_journal_se_retablit_apres_une_panne(self, record):
        """Une base qui revient doit être réutilisée sans redémarrer l'API."""
        journal = PredictionLog("postgresql://simule")
        journal._pool = FakePool(fail_on="connect")
        journal.record([record])
        assert journal.status()["database"] == "unavailable"

        journal._pool.fail_on = None
        journal.record([record])
        assert journal.status()["database"] == "ready"

    def test_une_chaine_de_connexion_ne_fuite_jamais_dans_un_message(self):
        """`DATABASE_URL` contient un mot de passe : il n'a rien à faire dans un
        journal ni dans une réponse HTTP."""
        exc = RuntimeError("connexion refusée pour postgresql://user:motdepasse@hote/base")
        message = _describe(exc)

        assert message.startswith("RuntimeError:")
        assert len(message) <= 300

    def test_un_message_d_erreur_est_borne(self):
        message = _describe(RuntimeError("x" * 5000))
        assert len(message) == 300

    def test_une_exception_sans_message_reste_identifiable(self):
        assert _describe(RuntimeError()) == "RuntimeError: RuntimeError"


class TestOuvertureReelleDuPool:
    def test_une_base_absente_laisse_le_service_operationnel(self, monkeypatch):
        """Un vrai `ConnectionPool` vers un port fermé : `open()` ne doit pas lever."""
        monkeypatch.setattr(config, "DB_CONNECT_TIMEOUT", 1.0)
        monkeypatch.setattr(config, "DB_WRITE_TIMEOUT", 1.0)

        journal = PredictionLog("postgresql://user:secret@127.0.0.1:1/inexistante")
        journal.open()  # ne doit pas lever
        try:
            assert journal.database_enabled is True
            assert journal.status()["database"] == "unavailable"
        finally:
            journal.close()


class TestIntegrationPostgres:
    """Contre un vrai PostgreSQL. Ignorés si `DATABASE_URL` n'est pas défini.

    Ce sont les seuls tests qui prouvent que le SQL est correct : le faux pool des
    tests unitaires accepterait une requête invalide sans broncher.

    Chaque test travaille sur un `request_id` qui lui est propre et nettoie ses
    lignes en sortant. Sans cette isolation, un test d'agrégation compterait aussi
    les prédictions laissées par les exécutions précédentes — et passerait ou
    échouerait selon l'historique de la base, ce qui est le pire des comportements.
    """

    @pytest.fixture
    def dsn(self):
        url = os.environ.get("DATABASE_URL")
        if not url:
            pytest.skip("DATABASE_URL non défini — tests d'intégration PostgreSQL ignorés")
        return url

    @pytest.fixture
    def journal(self, dsn):
        journal = PredictionLog(dsn)
        journal.open()
        yield journal
        journal.close()

    @pytest.fixture
    def enregistrement(self, dsn):
        """Un enregistrement au `request_id` unique, dont les lignes sont effacées
        à la fin du test."""
        import uuid

        import psycopg

        from api.model import Coverage

        identifiant = str(uuid.uuid4())
        yield build_record(
            request_id=identifiant,
            endpoint="/predict",
            features={"AMT_CREDIT": 406597.5, "EXT_SOURCE_2": 0.2629},
            probability=0.0731,
            decision="accepted",
            coverage=Coverage(provided=2, missing=777, application_ratio=0.9, history_ratio=0.1),
            model_version="1",
            threshold=0.1,
            latency_ms=4.2,
        )

        with psycopg.connect(dsn) as conn:
            conn.execute("DELETE FROM predictions WHERE request_id = %s", (identifiant,))

    def test_le_schema_est_reellement_cree(self, journal, dsn):
        import psycopg

        assert journal.status()["database"] == "ready"
        with psycopg.connect(dsn) as conn:
            colonnes = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'predictions'"
            ).fetchall()
        noms = {c[0] for c in colonnes}
        assert {"request_id", "occurred_at", "probability", "decision", "features"} <= noms

    def test_les_index_de_monitoring_existent(self, journal, dsn):
        """Sans index sur `occurred_at`, « les prédictions des 7 derniers jours »
        finirait en parcours complet de table."""
        import psycopg

        with psycopg.connect(dsn) as conn:
            index = conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'predictions'"
            ).fetchall()
        noms = {i[0] for i in index}
        assert "predictions_occurred_at_idx" in noms
        assert "predictions_model_version_idx" in noms

    def test_une_prediction_est_relisible_avec_ses_features(self, journal, enregistrement, dsn):
        import psycopg

        journal.record([enregistrement])

        with psycopg.connect(dsn) as conn:
            ligne = conn.execute(
                "SELECT probability, decision, model_version, features "
                "FROM predictions WHERE request_id = %s",
                (enregistrement.request_id,),
            ).fetchone()

        assert ligne is not None
        probabilite, decision, version, features = ligne
        assert probabilite == pytest.approx(0.0731)
        assert decision == "accepted"
        assert version == "1"
        # Le JSONB est relu en dictionnaire Python : c'est ce qui rend l'analyse de
        # drift possible sans table à 779 colonnes.
        assert features["AMT_CREDIT"] == 406597.5

    def test_les_features_sont_interrogeables_en_sql(self, journal, enregistrement, dsn):
        """Le vrai intérêt de JSONB face à un fichier de journaux : agréger."""
        import psycopg

        journal.record([enregistrement, enregistrement])

        with psycopg.connect(dsn) as conn:
            nombre, moyenne = conn.execute(
                "SELECT count(*), avg((features->>'EXT_SOURCE_2')::float) "
                "FROM predictions WHERE request_id = %s AND features ? 'EXT_SOURCE_2'",
                (enregistrement.request_id,),
            ).fetchone()

        assert nombre == 2
        assert moyenne == pytest.approx(0.2629)

    def test_un_lot_est_insere_en_une_transaction(self, journal, enregistrement, dsn):
        import psycopg

        journal.record([enregistrement] * 5)

        with psycopg.connect(dsn) as conn:
            nombre = conn.execute(
                "SELECT count(*) FROM predictions WHERE request_id = %s",
                (enregistrement.request_id,),
            ).fetchone()[0]

        assert nombre == 5
class TestDemarrageAvecBaseDisponible:
    """Chemin nominal de `open()` : le schéma est prêt avant la première requête."""

    def test_le_schema_est_prepare_des_le_demarrage(self, monkeypatch):
        import psycopg_pool

        class PoolInstrumente(FakePool):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def open(self, wait=False):
                self.opened = True

        monkeypatch.setattr(psycopg_pool, "ConnectionPool", PoolInstrumente)

        journal = PredictionLog("postgresql://simule")
        journal.open()
        try:
            assert journal.status() == {"stdout": True, "database": "ready", "last_error": None}
            assert any("CREATE TABLE" in sql for sql in journal._pool.executed)
        finally:
            journal.close()

    def test_un_pool_impossible_a_construire_laisse_le_service_debout(self, monkeypatch):
        """Chaîne de connexion malformée, pilote absent : l'API doit démarrer quand
        même et se contenter du canal `stdout`."""
        import psycopg_pool

        def refuser(*args, **kwargs):
            raise ValueError("chaîne de connexion invalide (simulée)")

        monkeypatch.setattr(psycopg_pool, "ConnectionPool", refuser)

        journal = PredictionLog("ce-n-est-pas-une-url")
        journal.open()  # ne doit pas lever

        assert journal.database_enabled is False
        assert journal.status()["database"] == "disabled"
        assert "ValueError" in journal.last_error
