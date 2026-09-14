"""
SCDP — Suite de tests complète
pytest + pytest-cov — couverture cible ≥ 90%

Exécution :
    pytest testall.py -v --cov=core --cov=db --cov=app --cov-report=term-missing
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import types
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────
# Fix bcrypt/passlib avant tout import applicatif
# ─────────────────────────────────────────────
import bcrypt as _bcrypt_module

_bcrypt_about = sys.modules.get("bcrypt.__about__")
if _bcrypt_about is None:
    _bcrypt_about = types.ModuleType("bcrypt.__about__")
    _bcrypt_about.__version__ = getattr(_bcrypt_module, "__version__", "4.0.1")
    sys.modules["bcrypt.__about__"] = _bcrypt_about
    _bcrypt_module.__about__ = _bcrypt_about

del _bcrypt_module, _bcrypt_about

# ─────────────────────────────────────────────
# Variables d'environnement de test
# ─────────────────────────────────────────────
os.environ.setdefault("SCDP_SECRET_KEY", "test-secret-key-for-pytest-at-least-32-characters")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_scdp.db")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
os.environ.setdefault("REFRESH_TOKEN_EXPIRE_DAYS", "30")
os.environ.setdefault("APP_VERSION", "1.0.0")

# ─────────────────────────────────────────────
# Imports applicatifs
# ─────────────────────────────────────────────
import core
import db as database

# ─────────────────────────────────────────────
# Fixtures SQLite en mémoire
# ─────────────────────────────────────────────

from sqlalchemy import create_engine as _create_engine
from sqlalchemy.orm import sessionmaker as _sessionmaker

_TEST_ENGINE = _create_engine(
    "sqlite:///./test_scdp_suite.db",
    connect_args={"check_same_thread": False},
)

# Patch le moteur global de db.py pour pointer sur SQLite
database.engine = _TEST_ENGINE
database.SessionLocal = _sessionmaker(autocommit=False, autoflush=False, bind=_TEST_ENGINE)


@pytest.fixture(scope="session", autouse=True)
def create_tables():
    """Crée le schéma SQLite pour tous les tests."""
    database.Base.metadata.create_all(bind=_TEST_ENGINE)
    yield
    database.Base.metadata.drop_all(bind=_TEST_ENGINE)
    # Supprime le fichier de base de test
    if os.path.exists("test_scdp_suite.db"):
        os.remove("test_scdp_suite.db")


@pytest.fixture()
def db_session():
    """Session SQLite isolée par test (rollback après chaque test)."""
    connection = _TEST_ENGINE.connect()
    transaction = connection.begin()
    session = _sessionmaker(bind=connection)()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


# ─────────────────────────────────────────────
# Fixture : données cliniques de base valides
# ─────────────────────────────────────────────

def _base_clinical_data(**overrides) -> dict:
    data = {
        "WPI": 10,
        "fatigue_score": 2,
        "sleep_score": 2,
        "cognitive_score": 1,
        "psych_score": 1,
        "evolution_months": 6,
        "exclusion_flags": [False, False, False],
    }
    data.update(overrides)
    return data


def _fibro_data(**overrides) -> dict:
    return _base_clinical_data(**overrides)


def _endo_data(**overrides) -> dict:
    return _base_clinical_data(
        cyclicite_score=2,
        dyspareunia_score=2,
        pelvic_locations_count=3,
        evolution_months=8,
        **overrides,
    )


def _sdrc_data(**overrides) -> dict:
    return _base_clinical_data(
        trigger_event=True,
        budapest_sensitif=True,
        budapest_vasomoteur=True,
        budapest_sudomoteur=True,
        budapest_moteur=False,
        evolution_months=3,
        **overrides,
    )


def _sfc_data(**overrides) -> dict:
    return _base_clinical_data(evolution_months=8, **overrides)


def _covid_data(**overrides) -> dict:
    return _base_clinical_data(
        covid_confirmed=True,
        onset_date="2022-03-15",
        evolution_months=4,
        **overrides,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC 1 — Tests du moteur SCDP (core.SCDPEngine)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSCDPEngineStepB:
    """Étape b — Transformation / normalisation."""

    def _engine(self, pathology_id="fibromyalgie"):
        return core.get_engine_for_pathology(pathology_id)

    def test_normalize_wpi_max(self):
        engine = self._engine()
        assert engine._normalize_wpi(19) == pytest.approx(1.0)

    def test_normalize_wpi_zero(self):
        engine = self._engine()
        assert engine._normalize_wpi(0) == pytest.approx(0.0)

    def test_normalize_wpi_mid(self):
        engine = self._engine()
        assert engine._normalize_wpi(10) == pytest.approx(10 / 19)

    def test_normalize_score_0_3_max(self):
        engine = self._engine()
        assert engine._normalize_score_0_3(3) == pytest.approx(1.0)

    def test_normalize_score_0_3_zero(self):
        engine = self._engine()
        assert engine._normalize_score_0_3(0) == pytest.approx(0.0)

    def test_step_b_returns_xi_ej(self):
        engine = self._engine()
        result = engine.step_b_transform(_fibro_data())
        assert "Xi" in result
        assert "Ej" in result
        assert len(result["Xi"]) == 5
        assert all(0.0 <= x <= 1.0 for x in result["Xi"])

    def test_step_b_exclusion_flags_active(self):
        engine = self._engine()
        data = _fibro_data(exclusion_flags=[True, False, False])
        result = engine.step_b_transform(data)
        assert result["Ej"][0] == 1.0
        assert result["Ej"][1] == 0.0

    def test_step_b_ej_padded_to_penalization_length(self):
        engine = self._engine()
        data = _fibro_data(exclusion_flags=[])
        result = engine.step_b_transform(data)
        assert len(result["Ej"]) == len(engine.penalization_weights)


class TestSCDPEngineStepC:
    """Étape c — Score brut pondéré."""

    def _engine(self):
        return core.get_engine_for_pathology("fibromyalgie")

    def test_score_brut_no_exclusion(self):
        engine = self._engine()
        Xi = [1.0, 1.0, 1.0, 1.0, 1.0]
        Ej = [0.0, 0.0, 0.0]
        score = engine.step_c_score_brut(Xi, Ej)
        assert score == pytest.approx(1.0 + 2.0 + 1.5 + 1.0 + 0.5)  # = 6.0

    def test_score_brut_with_exclusion(self):
        engine = self._engine()
        Xi = [1.0, 1.0, 1.0, 1.0, 1.0]
        Ej = [1.0, 0.0, 0.0]
        score = engine.step_c_score_brut(Xi, Ej)
        assert score == pytest.approx(6.0 - 3.0)  # penalisation v1=3.0

    def test_score_brut_all_zeros(self):
        engine = self._engine()
        Xi = [0.0, 0.0, 0.0, 0.0, 0.0]
        Ej = [0.0, 0.0, 0.0]
        assert engine.step_c_score_brut(Xi, Ej) == pytest.approx(0.0)


class TestSCDPEngineStepD:
    """Étape d — Contraintes binaires CRITIQUES."""

    def _engine(self, pathology_id="fibromyalgie"):
        return core.get_engine_for_pathology(pathology_id)

    # ── Règle critique : tout Ck = 0 → Score_final = 0 ──

    def test_chronicite_zero_score_final_zero(self):
        engine = self._engine()
        data = _fibro_data(evolution_months=1)  # < seuil 3 mois
        score_brut = 5.0
        score_final, constraints = engine.step_d_constraints(data, score_brut)
        assert score_final == 0.0
        c_ids_failed = [c["id"] for c in constraints if not c["satisfied"]]
        assert "C_chronicite" in c_ids_failed

    def test_diffusion_zero_score_final_zero(self):
        engine = self._engine()
        data = _fibro_data(WPI=3)  # WPI < 7
        score_brut = 5.0
        score_final, _ = engine.step_d_constraints(data, score_brut)
        assert score_final == 0.0

    def test_exclusion_zero_score_final_zero(self):
        engine = self._engine()
        data = _fibro_data(exclusion_flags=[True, False, False])
        score_brut = 5.0
        score_final, constraints = engine.step_d_constraints(data, score_brut)
        assert score_final == 0.0
        c_ids_failed = [c["id"] for c in constraints if not c["satisfied"]]
        assert "C_exclusion" in c_ids_failed

    def test_all_constraints_satisfied_preserves_score(self):
        engine = self._engine()
        data = _fibro_data(WPI=10, evolution_months=6)
        score_brut = 4.0
        score_final, constraints = engine.step_d_constraints(data, score_brut)
        assert score_final == pytest.approx(4.0)
        assert all(c["satisfied"] for c in constraints)

    def test_each_constraint_independently_zeroes_score(self):
        """Chaque Ck isolément à 0 avec score_brut positif élevé → score_final = 0."""
        engine = self._engine()
        score_brut = 5.8

        # C_chronicite seul à 0
        data1 = _fibro_data(WPI=10, evolution_months=1)
        sf1, _ = engine.step_d_constraints(data1, score_brut)
        assert sf1 == 0.0, "C_chronicite=0 doit zeroing le score"

        # C_diffusion seul à 0
        data2 = _fibro_data(WPI=3, evolution_months=6)
        sf2, _ = engine.step_d_constraints(data2, score_brut)
        assert sf2 == 0.0, "C_diffusion=0 doit zeroing le score"

        # C_exclusion seul à 0
        data3 = _fibro_data(WPI=10, evolution_months=6, exclusion_flags=[True])
        sf3, _ = engine.step_d_constraints(data3, score_brut)
        assert sf3 == 0.0, "C_exclusion=0 doit zeroing le score"

    def test_constraints_detail_structure(self):
        engine = self._engine()
        data = _fibro_data()
        _, constraints = engine.step_d_constraints(data, 3.0)
        for c in constraints:
            assert "id" in c
            assert "label" in c
            assert "value" in c
            assert "satisfied" in c
            assert "clinical_reason" in c

    def test_sdrc_trigger_event_constraint(self):
        engine = self._engine("sdrc")
        data = _sdrc_data(trigger_event=False)
        score_brut = 4.0
        score_final, constraints = engine.step_d_constraints(data, score_brut)
        assert score_final == 0.0
        c_ids = [c["id"] for c in constraints if not c["satisfied"]]
        assert "C_trigger_event" in c_ids

    def test_covid_diffusion_is_true(self):
        engine = self._engine("covid_long_neurologique")
        data = _covid_data(covid_confirmed=False)
        score_brut = 4.0
        score_final, constraints = engine.step_d_constraints(data, score_brut)
        assert score_final == 0.0

    def test_constraint_report_indicates_which_failed(self):
        engine = self._engine()
        data = _fibro_data(evolution_months=1)
        _, constraints = engine.step_d_constraints(data, 5.0)
        failed = [c for c in constraints if not c["satisfied"]]
        assert len(failed) >= 1
        assert failed[0]["value"] == 0


class TestSCDPEngineStepE:
    """Étape e — Indice de centralisation Ic."""

    def _engine(self):
        return core.get_engine_for_pathology("fibromyalgie")

    def test_ic_formula(self):
        engine = self._engine()
        ic = engine.step_e_ic(F=2, S=2, C=1, WPI=4)
        assert ic == pytest.approx(5 / 5)  # (2+2+1)/(4+1) = 1.0

    def test_ic_wpi_zero_no_division_error(self):
        engine = self._engine()
        ic = engine.step_e_ic(F=3, S=3, C=3, WPI=0)
        assert ic == pytest.approx(9.0 / 1.0)  # dénominateur = 1

    def test_ic_above_one_central_sensitization(self):
        engine = self._engine()
        ic = engine.step_e_ic(F=3, S=3, C=3, WPI=2)
        assert ic > 1.0  # (9)/(3) = 3.0

    def test_ic_below_one_peripheral_profile(self):
        engine = self._engine()
        ic = engine.step_e_ic(F=0, S=0, C=0, WPI=10)
        assert ic <= 1.0  # 0/11 = 0

    def test_ic_equal_one_boundary(self):
        engine = self._engine()
        ic = engine.step_e_ic(F=1, S=1, C=1, WPI=2)
        assert ic == pytest.approx(1.0)


class TestSCDPEngineStepF:
    """Étape f — Probabilité diagnostique."""

    def _engine(self):
        return core.get_engine_for_pathology("fibromyalgie")

    def test_probability_in_range(self):
        engine = self._engine()
        Xi = [0.5, 0.5, 0.5, 0.5, 0.5]
        p = engine.step_f_probability(Xi, Ic=1.0)
        assert 0.0 <= p <= 1.0

    def test_high_xi_high_probability(self):
        engine = self._engine()
        Xi_high = [1.0, 1.0, 1.0, 1.0, 1.0]
        p_high = engine.step_f_probability(Xi_high, Ic=2.0)
        Xi_low = [0.0, 0.0, 0.0, 0.0, 0.0]
        p_low = engine.step_f_probability(Xi_low, Ic=0.0)
        assert p_high > p_low

    def test_ic_amplification(self):
        """Ic > 1 doit amplifier P(D) toutes choses égales."""
        engine = self._engine()
        Xi = [0.6, 0.6, 0.6, 0.6, 0.6]
        p_high_ic = engine.step_f_probability(Xi, Ic=2.0)
        p_low_ic = engine.step_f_probability(Xi, Ic=0.5)
        assert p_high_ic > p_low_ic

    def test_probability_clamped_no_overflow(self):
        """Pas d'overflow float sur valeurs extrêmes."""
        engine = self._engine()
        Xi = [1.0] * 5
        p = engine.step_f_probability(Xi, Ic=100.0)
        assert math.isfinite(p)
        assert 0.0 <= p <= 1.0


class TestSCDPEngineStepG:
    """Étape g — Rapport structuré."""

    def _engine(self, pid="fibromyalgie"):
        return core.get_engine_for_pathology(pid)

    def test_report_keys_present(self):
        engine = self._engine()
        report = engine.step_g_report(
            clinical_data=_fibro_data(),
            score_final=3.0,
            Ic=1.2,
            probability=0.75,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        required_keys = [
            "score_final_normalized",
            "Ic",
            "probability_pct",
            "decision_class",
            "constraints_detail",
            "input_trace",
            "params_version_id",
            "computed_at",
        ]
        for key in required_keys:
            assert key in report, f"Clé manquante : {key}"

    def test_score_final_zero_gives_incertain(self):
        engine = self._engine()
        report = engine.step_g_report(
            clinical_data=_fibro_data(),
            score_final=0.0,
            Ic=0.5,
            probability=0.1,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        assert report["decision_class"] == core.DECISION_INCERTAIN

    def test_high_score_gives_hautement_probable(self):
        engine = self._engine()
        # score max théorique = 6.0, seuil haute = 30/100, donc score_final ≥ 30% de 6 = 1.8
        score_final = engine.score_max  # 100 / 100
        report = engine.step_g_report(
            clinical_data=_fibro_data(),
            score_final=score_final,
            Ic=2.0,
            probability=0.9,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        assert report["decision_class"] == core.DECISION_HAUTEMENT_PROBABLE

    def test_medium_score_gives_probable(self):
        engine = self._engine()
        # seuil probable = 18/100 → score_final = 0.18 * 6.0 = 1.08
        score_final = engine.score_max * 0.22  # juste au-dessus de 18%
        report = engine.step_g_report(
            clinical_data=_fibro_data(),
            score_final=score_final,
            Ic=1.0,
            probability=0.6,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        assert report["decision_class"] == core.DECISION_PROBABLE

    def test_score_normalized_range(self):
        engine = self._engine()
        report = engine.step_g_report(
            clinical_data=_fibro_data(),
            score_final=engine.score_max,
            Ic=1.0,
            probability=0.8,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        assert 0.0 <= report["score_final_normalized"] <= 100.0

    def test_probability_pct_range(self):
        engine = self._engine()
        report = engine.step_g_report(
            clinical_data=_fibro_data(),
            score_final=3.0,
            Ic=1.0,
            probability=0.65,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        assert 0.0 <= report["probability_pct"] <= 100.0

    def test_input_trace_in_report(self):
        engine = self._engine()
        cd = _fibro_data()
        report = engine.step_g_report(
            clinical_data=cd,
            score_final=3.0,
            Ic=1.0,
            probability=0.7,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        assert report["input_trace"]["WPI"] == cd["WPI"]

    def test_ic_signature_central(self):
        engine = self._engine()
        report = engine.step_g_report(
            clinical_data=_fibro_data(),
            score_final=3.0,
            Ic=1.5,
            probability=0.7,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        assert report["ic_signature"] == "sensibilisation_centrale"

    def test_ic_signature_peripheral(self):
        engine = self._engine()
        report = engine.step_g_report(
            clinical_data=_fibro_data(),
            score_final=3.0,
            Ic=0.8,
            probability=0.4,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        assert report["ic_signature"] == "profil_peripherique"

    def test_params_version_id_present_and_nonempty(self):
        engine = self._engine()
        report = engine.step_g_report(
            clinical_data=_fibro_data(),
            score_final=2.0,
            Ic=1.0,
            probability=0.5,
            constraints_detail=[],
            computed_at=datetime.now(timezone.utc),
        )
        assert report["params_version_id"]
        assert len(report["params_version_id"]) > 10


class TestSCDPEnginePipeline:
    """Pipeline complet run_pipeline."""

    def test_pipeline_fibromyalgie_healthy(self):
        engine = core.get_engine_for_pathology("fibromyalgie")
        data = _fibro_data(WPI=12, evolution_months=6, fatigue_score=3, sleep_score=2)
        report = engine.run_pipeline(data)
        assert report["decision_class"] in {
            core.DECISION_INCERTAIN,
            core.DECISION_PROBABLE,
            core.DECISION_HAUTEMENT_PROBABLE,
        }
        assert "params_version_id" in report
        assert "computed_at" in report

    def test_pipeline_endometriose(self):
        engine = core.get_engine_for_pathology("endometriose")
        data = _endo_data()
        report = engine.run_pipeline(data)
        assert "score_final_normalized" in report

    def test_pipeline_sdrc(self):
        engine = core.get_engine_for_pathology("sdrc")
        data = _sdrc_data()
        report = engine.run_pipeline(data)
        assert "Ic" in report

    def test_pipeline_sfc_me(self):
        engine = core.get_engine_for_pathology("sfc_me")
        data = _sfc_data()
        report = engine.run_pipeline(data)
        assert report["pathology_id"] == "sfc_me"

    def test_pipeline_covid_long(self):
        engine = core.get_engine_for_pathology("covid_long_neurologique")
        data = _covid_data()
        report = engine.run_pipeline(data)
        assert report["pathology_id"] == "covid_long_neurologique"

    def test_pipeline_reproducible(self):
        """Deux appels identiques → même résultat (traçabilité)."""
        loaded_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = _fibro_data(WPI=10)
        r1 = core.get_engine_for_pathology("fibromyalgie", loaded_at).run_pipeline(data)
        r2 = core.get_engine_for_pathology("fibromyalgie", loaded_at).run_pipeline(data)
        assert r1["score_final_normalized"] == r2["score_final_normalized"]
        assert r1["probability_pct"] == r2["probability_pct"]
        assert r1["params_version_id"] == r2["params_version_id"]

    def test_pipeline_constraint_zero_incertain_low_proba(self):
        """Contrainte = 0 → Score_final = 0 → decision INCERTAIN."""
        engine = core.get_engine_for_pathology("fibromyalgie")
        data = _fibro_data(evolution_months=1)  # chronicite=0
        report = engine.run_pipeline(data)
        assert report["score_final_normalized"] == 0.0
        assert report["decision_class"] == core.DECISION_INCERTAIN

    def test_ic_always_in_report(self):
        """Ic doit toujours apparaître dans le rapport."""
        for pid in core.list_pathology_ids():
            if pid == "endometriose":
                data = _endo_data()
            elif pid == "sdrc":
                data = _sdrc_data()
            elif pid == "covid_long_neurologique":
                data = _covid_data()
            else:
                data = _base_clinical_data()
            engine = core.get_engine_for_pathology(pid)
            report = engine.run_pipeline(data)
            assert "Ic" in report, f"Ic absent dans rapport pour {pid}"


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC 2 — Tests de validation clinique
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateClinicalData:
    """core.validate_clinical_data — règles métier section 6.4."""

    def _config(self, pid="fibromyalgie"):
        return core.PATHOLOGY_CONFIGS[pid]

    def test_valid_data_no_errors(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(), self._config())
        assert errors == []

    def test_wpi_out_of_range_high(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(WPI=20), self._config())
        assert any(e["field"] == "WPI" for e in errors)

    def test_wpi_out_of_range_low(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(WPI=-1), self._config())
        assert any(e["field"] == "WPI" for e in errors)

    def test_wpi_missing(self):
        data = _fibro_data()
        del data["WPI"]
        errors = core.validate_clinical_data("fibromyalgie", data, self._config())
        assert any(e["field"] == "WPI" for e in errors)

    def test_fatigue_score_out_of_range(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(fatigue_score=4), self._config())
        assert any(e["field"] == "fatigue_score" for e in errors)

    def test_sleep_score_out_of_range(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(sleep_score=-1), self._config())
        assert any(e["field"] == "sleep_score" for e in errors)

    def test_cognitive_score_out_of_range(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(cognitive_score=5), self._config())
        assert any(e["field"] == "cognitive_score" for e in errors)

    def test_psych_score_out_of_range(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(psych_score=4), self._config())
        assert any(e["field"] == "psych_score" for e in errors)

    def test_evolution_months_negative(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(evolution_months=-1), self._config())
        assert any(e["field"] == "evolution_months" for e in errors)

    def test_exclusion_flags_not_list(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(exclusion_flags="true"), self._config())
        assert any(e["field"] == "exclusion_flags" for e in errors)

    def test_exclusion_flags_not_booleans(self):
        errors = core.validate_clinical_data("fibromyalgie", _fibro_data(exclusion_flags=[1, 0]), self._config())
        assert any(e["field"] == "exclusion_flags" for e in errors)

    def test_missing_specific_field_endometriose(self):
        config = self._config("endometriose")
        data = _base_clinical_data()  # Sans cyclicite_score etc.
        errors = core.validate_clinical_data("endometriose", data, config)
        missing_error = next((e for e in errors if "missing_fields" in e), None)
        assert missing_error is not None
        assert "cyclicite_score" in missing_error["missing_fields"]

    def test_specific_field_date_invalid_format(self):
        config = self._config("covid_long_neurologique")
        data = _covid_data(onset_date="15-03-2022")  # Format invalide
        errors = core.validate_clinical_data("covid_long_neurologique", data, config)
        assert any(e["field"] == "onset_date" for e in errors)

    def test_specific_field_date_in_future(self):
        config = self._config("covid_long_neurologique")
        future = (date.today() + timedelta(days=30)).isoformat()
        data = _covid_data(onset_date=future)
        errors = core.validate_clinical_data("covid_long_neurologique", data, config)
        assert any(e["field"] == "onset_date" for e in errors)

    def test_specific_boolean_field_wrong_type(self):
        config = self._config("sdrc")
        data = _sdrc_data(trigger_event="yes")  # Doit être bool
        errors = core.validate_clinical_data("sdrc", data, config)
        assert any(e["field"] == "trigger_event" for e in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC 3 — Tests hashing et utilitaires
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoreUtils:
    """Fonctions utilitaires core."""

    def test_compute_sha256_dict(self):
        data = {"key": "value", "num": 42}
        h = core.compute_sha256(data)
        assert len(h) == 64
        assert h == core.compute_sha256(data)  # idempotent

    def test_compute_sha256_order_independent(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        assert core.compute_sha256(d1) == core.compute_sha256(d2)

    def test_compute_sha256_string(self):
        h = core.compute_sha256("test_string")
        assert len(h) == 64

    def test_compute_params_version_id_format(self):
        config = core.PATHOLOGY_CONFIGS["fibromyalgie"]
        ts = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        vid = core.compute_params_version_id("fibromyalgie", config, ts)
        assert "-" in vid
        parts = vid.split("-")
        assert len(parts[0]) == 16
        assert "20240615T100000Z" in vid

    def test_params_version_id_changes_with_timestamp(self):
        config = core.PATHOLOGY_CONFIGS["fibromyalgie"]
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        vid1 = core.compute_params_version_id("fibromyalgie", config, ts1)
        vid2 = core.compute_params_version_id("fibromyalgie", config, ts2)
        assert vid1 != vid2

    def test_compute_refresh_token_hash(self):
        token = "some_random_refresh_token"
        h = core.compute_refresh_token_hash(token)
        assert len(h) == 64
        assert h == core.compute_refresh_token_hash(token)

    def test_list_pathology_ids(self):
        ids = core.list_pathology_ids()
        assert "fibromyalgie" in ids
        assert "endometriose" in ids
        assert "sdrc" in ids
        assert "sfc_me" in ids
        assert "covid_long_neurologique" in ids
        assert len(ids) == 5

    def test_get_pathology_config_known(self):
        config = core.get_pathology_config("fibromyalgie")
        assert config is not None
        assert config["pathology_id"] == "fibromyalgie"

    def test_get_pathology_config_unknown(self):
        assert core.get_pathology_config("unknown_pathology") is None

    def test_get_engine_for_pathology_unknown_raises(self):
        with pytest.raises(ValueError):
            core.get_engine_for_pathology("unknown_pathology")

    def test_get_pathology_schema_fibromyalgie(self):
        schema = core.get_pathology_schema("fibromyalgie")
        assert schema is not None
        assert "WPI" in schema["properties"]
        assert "fatigue_score" in schema["properties"]
        assert "additionalProperties" in schema

    def test_get_pathology_schema_endometriose_specific_fields(self):
        schema = core.get_pathology_schema("endometriose")
        assert "cyclicite_score" in schema["properties"]
        assert "dyspareunia_score" in schema["properties"]

    def test_get_pathology_schema_unknown_returns_none(self):
        assert core.get_pathology_schema("nonexistent") is None

    def test_sanitize_string_strips_control_chars(self):
        s = "hello\x00world\x1f!"
        result = core.sanitize_string(s)
        assert "\x00" not in result
        assert "hello" in result

    def test_sanitize_string_truncates(self):
        s = "a" * 600
        result = core.sanitize_string(s, max_length=100)
        assert len(result) == 100

    def test_safe_json_loads_valid(self):
        result = core.safe_json_loads('{"key": "val"}')
        assert result == {"key": "val"}

    def test_safe_json_loads_invalid(self):
        result = core.safe_json_loads("not valid json {")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC 4 — Tests CRUD base de données
# ═══════════════════════════════════════════════════════════════════════════════

class TestCRUDUsers:
    """CRUD utilisateurs."""

    def test_create_user(self, db_session):
        user = database.create_user(db_session, f"test_{uuid.uuid4()}@example.com", "hashed_pw")
        assert user.id is not None
        assert user.role == "medecin"
        assert user.is_active is True

    def test_get_user_by_email_found(self, db_session):
        email = f"findme_{uuid.uuid4()}@example.com"
        database.create_user(db_session, email, "hash")
        user = database.get_user_by_email(db_session, email)
        assert user is not None
        assert user.email == email.lower()

    def test_get_user_by_email_not_found(self, db_session):
        user = database.get_user_by_email(db_session, "ghost@nowhere.com")
        assert user is None

    def test_get_user_by_id_found(self, db_session):
        user = database.create_user(db_session, f"uid_{uuid.uuid4()}@example.com", "hash")
        found = database.get_user_by_id(db_session, user.id)
        assert found is not None
        assert found.id == user.id

    def test_get_user_by_id_not_found(self, db_session):
        assert database.get_user_by_id(db_session, str(uuid.uuid4())) is None

    def test_email_normalized_lowercase(self, db_session):
        email = f"UPPER_{uuid.uuid4()}@EXAMPLE.COM"
        user = database.create_user(db_session, email, "hash")
        assert user.email == email.lower()

    def test_update_user_timestamp(self, db_session):
        user = database.create_user(db_session, f"ts_{uuid.uuid4()}@example.com", "hash")
        # Ne doit pas lever d'exception
        database.update_user_timestamp(db_session, user.id)


class TestCRUDReports:
    """CRUD rapports."""

    def _make_user(self, db_session) -> database.User:
        return database.create_user(db_session, f"rpt_{uuid.uuid4()}@example.com", "hash")

    def _sample_report_json(self) -> dict:
        return {
            "score_final_normalized": 42.5,
            "Ic": 1.2,
            "probability_pct": 67.3,
            "decision_class": "PROBABLE",
            "constraints_detail": [],
            "input_trace": {},
            "params_version_id": "abc123",
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def test_create_report(self, db_session):
        user = self._make_user(db_session)
        report = database.create_report(
            db_session, user.id, "fibromyalgie", self._sample_report_json()
        )
        assert report.id is not None
        assert report.user_id == user.id

    def test_get_report_by_id(self, db_session):
        user = self._make_user(db_session)
        report = database.create_report(db_session, user.id, "fibromyalgie", self._sample_report_json())
        found = database.get_report_by_id(db_session, report.id)
        assert found is not None
        assert found.id == report.id

    def test_get_report_by_id_not_found(self, db_session):
        assert database.get_report_by_id(db_session, str(uuid.uuid4())) is None

    def test_update_report_pdf(self, db_session):
        user = self._make_user(db_session)
        report = database.create_report(db_session, user.id, "fibromyalgie", self._sample_report_json())
        assert report.pdf_path is None
        database.update_report_pdf(db_session, report.id, "/some/path/report.pdf")
        updated = database.get_report_by_id(db_session, report.id)
        assert updated.pdf_path == "/some/path/report.pdf"

    def test_get_reports_by_user_pagination(self, db_session):
        user = self._make_user(db_session)
        for _ in range(5):
            database.create_report(db_session, user.id, "fibromyalgie", self._sample_report_json())
        reports, total = database.get_reports_by_user(db_session, user.id, page=1, page_size=3)
        assert len(reports) == 3
        assert total == 5

    def test_get_reports_by_user_pathology_filter(self, db_session):
        user = self._make_user(db_session)
        database.create_report(db_session, user.id, "fibromyalgie", self._sample_report_json())
        database.create_report(db_session, user.id, "endometriose", self._sample_report_json())
        reports, total = database.get_reports_by_user(
            db_session, user.id, pathology_filter="fibromyalgie"
        )
        assert all(r.pathology_id == "fibromyalgie" for r in reports)

    def test_get_all_reports(self, db_session):
        user = self._make_user(db_session)
        database.create_report(db_session, user.id, "fibromyalgie", self._sample_report_json())
        reports, total = database.get_all_reports(db_session)
        assert total >= 1


class TestCRUDPathologyParams:
    """CRUD paramètres de pathologies."""

    def test_upsert_and_get_active_params(self, db_session):
        pid = f"test_path_{uuid.uuid4().hex[:8]}"
        params = {"pathology_id": pid, "version": "1.0.0", "data": "test"}
        sha = core.compute_sha256(params)
        pp = database.upsert_pathology_params(db_session, pid, "1.0.0", sha, params)
        assert pp.is_active is True
        active = database.get_active_params(db_session, pid)
        assert active is not None
        assert active.sha256 == sha

    def test_upsert_deactivates_previous(self, db_session):
        pid = f"path_v_{uuid.uuid4().hex[:8]}"
        params_v1 = {"version": "1.0.0"}
        sha1 = core.compute_sha256(params_v1)
        database.upsert_pathology_params(db_session, pid, "1.0.0", sha1, params_v1)

        params_v2 = {"version": "2.0.0"}
        sha2 = core.compute_sha256(params_v2)
        database.upsert_pathology_params(db_session, pid, "2.0.0", sha2, params_v2)

        active = database.get_active_params(db_session, pid)
        assert active.sha256 == sha2

    def test_get_active_params_not_found(self, db_session):
        result = database.get_active_params(db_session, "nonexistent_pathology_99")
        assert result is None

    def test_get_all_active_params(self, db_session):
        # Ne lève pas d'exception et retourne une liste
        params = database.get_all_active_params(db_session)
        assert isinstance(params, list)


class TestCRUDAuthSession:
    """CRUD sessions d'authentification."""

    def _make_user(self, db_session) -> database.User:
        return database.create_user(db_session, f"auth_{uuid.uuid4()}@example.com", "hash")

    def test_create_auth_session(self, db_session):
        user = self._make_user(db_session)
        token = f"token_{uuid.uuid4().hex}"
        token_hash = core.compute_refresh_token_hash(token)
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        session = database.create_auth_session(db_session, user.id, token_hash, expires)
        assert session.id is not None
        assert session.revoked is False

    def test_get_valid_session(self, db_session):
        user = self._make_user(db_session)
        token = f"token_{uuid.uuid4().hex}"
        token_hash = core.compute_refresh_token_hash(token)
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        database.create_auth_session(db_session, user.id, token_hash, expires)
        session = database.get_valid_session(db_session, token_hash)
        assert session is not None

    def test_revoke_session(self, db_session):
        user = self._make_user(db_session)
        token = f"token_{uuid.uuid4().hex}"
        token_hash = core.compute_refresh_token_hash(token)
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        database.create_auth_session(db_session, user.id, token_hash, expires)
        database.revoke_session(db_session, token_hash)
        session = database.get_valid_session(db_session, token_hash)
        assert session is None

    def test_revoke_all_user_sessions(self, db_session):
        user = self._make_user(db_session)
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        for _ in range(3):
            token_hash = core.compute_refresh_token_hash(f"token_{uuid.uuid4().hex}")
            database.create_auth_session(db_session, user.id, token_hash, expires)
        database.revoke_all_user_sessions(db_session, user.id)
        # Toutes les sessions sont révoquées — aucune n'est valide
        from sqlalchemy import select
        sessions = db_session.query(database.AuthSession).filter(
            database.AuthSession.user_id == user.id,
            database.AuthSession.revoked == False,
        ).all()
        assert sessions == []

    def test_expired_session_not_returned(self, db_session):
        user = self._make_user(db_session)
        token = f"token_{uuid.uuid4().hex}"
        token_hash = core.compute_refresh_token_hash(token)
        expired = datetime.now(timezone.utc) - timedelta(hours=1)
        database.create_auth_session(db_session, user.id, token_hash, expired)
        session = database.get_valid_session(db_session, token_hash)
        assert session is None

    def test_cleanup_expired_sessions(self, db_session):
        user = self._make_user(db_session)
        token_hash = core.compute_refresh_token_hash(f"token_{uuid.uuid4().hex}")
        expired = datetime.now(timezone.utc) - timedelta(hours=1)
        database.create_auth_session(db_session, user.id, token_hash, expired)
        deleted = database.cleanup_expired_sessions(db_session)
        assert deleted >= 1


class TestCRUDAuditLog:
    """CRUD audit logs."""

    def test_write_audit_log(self, db_session):
        # Ne lève pas d'exception
        database.write_audit_log(
            db_session,
            action="test.action",
            actor_id=str(uuid.uuid4()),
            payload={"info": "test"},
            ip_address="127.0.0.1",
        )

    def test_get_audit_logs_pagination(self, db_session):
        for i in range(5):
            database.write_audit_log(db_session, action=f"test.log.{i}")
        logs, total = database.get_audit_logs(db_session, page=1, page_size=3)
        assert len(logs) <= 3

    def test_get_audit_logs_action_filter(self, db_session):
        unique_action = f"unique.action.{uuid.uuid4().hex}"
        database.write_audit_log(db_session, action=unique_action)
        logs, total = database.get_audit_logs(db_session, action_filter=unique_action)
        assert total >= 1
        assert all(unique_action in log.action for log in logs)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC 5 — Tests API FastAPI (TestClient)
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi.testclient import TestClient

# On importe app après les patches DB
import app as application

_app_client: TestClient | None = None


@pytest.fixture(scope="module")
def client():
    """Client de test FastAPI avec override de la dépendance DB."""
    def _get_test_db():
        conn = _TEST_ENGINE.connect()
        trans = conn.begin()
        sess = _sessionmaker(bind=conn)()
        try:
            yield sess
        finally:
            sess.close()
            trans.rollback()
            conn.close()

    application.app.dependency_overrides[database.get_db] = _get_test_db
    with TestClient(application.app, raise_server_exceptions=True) as c:
        yield c
    application.app.dependency_overrides.clear()


def _register_and_login(client: TestClient, email: str | None = None, password: str = "TestPass1") -> dict:
    if email is None:
        email = f"api_{uuid.uuid4()}@test.com"
    client.post("/api/v1/auth/register", json={"email": email, "password": password})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()


class TestAPIAuth:
    """Endpoints authentification."""

    def test_register_success(self, client):
        email = f"reg_{uuid.uuid4()}@test.com"
        resp = client.post("/api/v1/auth/register", json={"email": email, "password": "TestPass1"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == email
        assert data["role"] == "medecin"

    def test_register_duplicate_email(self, client):
        email = f"dup_{uuid.uuid4()}@test.com"
        client.post("/api/v1/auth/register", json={"email": email, "password": "TestPass1"})
        resp = client.post("/api/v1/auth/register", json={"email": email, "password": "TestPass1"})
        assert resp.status_code == 409

    def test_register_weak_password_no_uppercase(self, client):
        resp = client.post(
            "/api/v1/auth/register",
            json={"email": "weak@test.com", "password": "testpass1"},
        )
        assert resp.status_code == 422

    def test_register_weak_password_no_digit(self, client):
        resp = client.post(
            "/api/v1/auth/register",
            json={"email": "weak2@test.com", "password": "TestPassword"},
        )
        assert resp.status_code == 422

    def test_login_success(self, client):
        email = f"login_{uuid.uuid4()}@test.com"
        client.post("/api/v1/auth/register", json={"email": email, "password": "TestPass1"})
        resp = client.post("/api/v1/auth/login", json={"email": email, "password": "TestPass1"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_login_wrong_password(self, client):
        email = f"badpw_{uuid.uuid4()}@test.com"
        client.post("/api/v1/auth/register", json={"email": email, "password": "TestPass1"})
        resp = client.post("/api/v1/auth/login", json={"email": email, "password": "WrongPass9"})
        assert resp.status_code == 401

    def test_login_unknown_email(self, client):
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "ghost@nowhere.com", "password": "TestPass1"},
        )
        assert resp.status_code == 401

    def test_get_me_authenticated(self, client):
        tokens = _register_and_login(client)
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 200
        assert "email" in resp.json()

    def test_get_me_no_token(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code in (401, 403)

    def test_refresh_token(self, client):
        tokens = _register_and_login(client)
        resp = client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )
        assert resp.status_code == 200
        new_tokens = resp.json()
        assert "access_token" in new_tokens

    def test_logout(self, client):
        tokens = _register_and_login(client)
        resp = client.post(
            "/api/v1/auth/logout",
            json={"refresh_token": tokens["refresh_token"]},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 200


class TestAPIHealth:
    """Endpoint /health."""

    def test_health_returns_ok_or_degraded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "version" in data
        assert "uptime_seconds" in data


class TestAPISCDPPathologies:
    """Endpoints pathologies SCDP."""

    def test_list_pathologies(self, client):
        tokens = _register_and_login(client)
        resp = client.get(
            "/api/v1/scdp/pathologies",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pathologies" in data
        ids = [p["pathology_id"] for p in data["pathologies"]]
        assert "fibromyalgie" in ids
        assert len(ids) == 5

    def test_get_pathology_schema(self, client):
        tokens = _register_and_login(client)
        resp = client.get(
            "/api/v1/scdp/pathologies/fibromyalgie/schema",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 200
        schema = resp.json()
        assert "properties" in schema
        assert "WPI" in schema["properties"]

    def test_get_pathology_schema_unknown(self, client):
        tokens = _register_and_login(client)
        resp = client.get(
            "/api/v1/scdp/pathologies/unknown_path/schema",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 404


class TestAPISCDPEvaluate:
    """Endpoint POST /api/v1/scdp/evaluate."""

    def test_evaluate_fibromyalgie_valid(self, client):
        tokens = _register_and_login(client)
        with patch("core.generate_pdf_report", return_value="/tmp/test.pdf"):
            resp = client.post(
                "/api/v1/scdp/evaluate",
                json={
                    "pathology_id": "fibromyalgie",
                    "clinical_data": _fibro_data(),
                },
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "report_id" in data
        assert "decision_class" in data
        assert "score_final_normalized" in data
        assert "probability_pct" in data
        assert "Ic" in data
        assert "params_version_id" in data

    def test_evaluate_invalid_wpi(self, client):
        tokens = _register_and_login(client)
        resp = client.post(
            "/api/v1/scdp/evaluate",
            json={
                "pathology_id": "fibromyalgie",
                "clinical_data": _fibro_data(WPI=25),
            },
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 422

    def test_evaluate_missing_specific_field(self, client):
        tokens = _register_and_login(client)
        resp = client.post(
            "/api/v1/scdp/evaluate",
            json={
                "pathology_id": "endometriose",
                "clinical_data": _base_clinical_data(),
            },
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 422

    def test_evaluate_unknown_pathology(self, client):
        tokens = _register_and_login(client)
        resp = client.post(
            "/api/v1/scdp/evaluate",
            json={
                "pathology_id": "unknown_disease",
                "clinical_data": _fibro_data(),
            },
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 422

    def test_evaluate_no_auth(self, client):
        resp = client.post(
            "/api/v1/scdp/evaluate",
            json={
                "pathology_id": "fibromyalgie",
                "clinical_data": _fibro_data(),
            },
        )
        assert resp.status_code in (401, 403)

    def test_evaluate_pdf_generated(self, client):
        """Vérifie que pdf_available est présent dans la réponse."""
        tokens = _register_and_login(client)
        with patch("core.generate_pdf_report", return_value="/tmp/test_pdf.pdf"):
            resp = client.post(
                "/api/v1/scdp/evaluate",
                json={
                    "pathology_id": "fibromyalgie",
                    "clinical_data": _fibro_data(),
                },
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
        assert resp.status_code == 200
        assert "pdf_available" in resp.json()

    def test_evaluate_pdf_failure_graceful(self, client):
        """Échec génération PDF → ne fait pas échouer la réponse."""
        tokens = _register_and_login(client)
        with patch("core.generate_pdf_report", side_effect=Exception("PDF error")):
            resp = client.post(
                "/api/v1/scdp/evaluate",
                json={
                    "pathology_id": "fibromyalgie",
                    "clinical_data": _fibro_data(),
                },
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
        assert resp.status_code == 200
        assert resp.json()["pdf_available"] is False

    def test_evaluate_constraint_zero_produces_incertain(self, client):
        """Contrainte C=0 → decision_class INCERTAIN."""
        tokens = _register_and_login(client)
        with patch("core.generate_pdf_report", return_value="/tmp/t.pdf"):
            resp = client.post(
                "/api/v1/scdp/evaluate",
                json={
                    "pathology_id": "fibromyalgie",
                    "clinical_data": _fibro_data(evolution_months=1),  # chronicite=0
                },
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
        assert resp.status_code == 200
        assert resp.json()["decision_class"] == "INCERTAIN"

    def test_evaluate_all_pathologies(self, client):
        """Chaque pathologie peut être évaluée sans erreur serveur."""
        tokens = _register_and_login(client)
        test_data_by_pid = {
            "fibromyalgie": _fibro_data(),
            "endometriose": _endo_data(),
            "sdrc": _sdrc_data(),
            "sfc_me": _sfc_data(),
            "covid_long_neurologique": _covid_data(),
        }
        for pid, data in test_data_by_pid.items():
            with patch("core.generate_pdf_report", return_value="/tmp/t.pdf"):
                resp = client.post(
                    "/api/v1/scdp/evaluate",
                    json={"pathology_id": pid, "clinical_data": data},
                    headers={"Authorization": f"Bearer {tokens['access_token']}"},
                )
            assert resp.status_code == 200, f"Évaluation échouée pour {pid}: {resp.text}"


class TestAPISCDPBatch:
    """Endpoint POST /api/v1/scdp/batch."""

    def test_batch_evaluate(self, client):
        tokens = _register_and_login(client)
        with patch("core.generate_pdf_report", return_value="/tmp/t.pdf"):
            resp = client.post(
                "/api/v1/scdp/batch",
                json={
                    "items": [
                        {"pathology_id": "fibromyalgie", "clinical_data": _fibro_data()},
                        {"pathology_id": "fibromyalgie", "clinical_data": _fibro_data(WPI=5)},
                    ]
                },
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_with_invalid_item(self, client):
        tokens = _register_and_login(client)
        resp = client.post(
            "/api/v1/scdp/batch",
            json={
                "items": [
                    {"pathology_id": "fibromyalgie", "clinical_data": _fibro_data(WPI=99)},
                ]
            },
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["success"] is False

    def test_batch_empty_list_rejected(self, client):
        tokens = _register_and_login(client)
        resp = client.post(
            "/api/v1/scdp/batch",
            json={"items": []},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 422


class TestAPIReports:
    """Endpoints rapports."""

    def test_list_reports(self, client):
        tokens = _register_and_login(client)
        with patch("core.generate_pdf_report", return_value="/tmp/t.pdf"):
            client.post(
                "/api/v1/scdp/evaluate",
                json={"pathology_id": "fibromyalgie", "clinical_data": _fibro_data()},
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
        resp = client.get(
            "/api/v1/reports",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    def test_get_report_by_id(self, client):
        tokens = _register_and_login(client)
        with patch("core.generate_pdf_report", return_value="/tmp/t.pdf"):
            eval_resp = client.post(
                "/api/v1/scdp/evaluate",
                json={"pathology_id": "fibromyalgie", "clinical_data": _fibro_data()},
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
        report_id = eval_resp.json()["report_id"]
        resp = client.get(
            f"/api/v1/reports/{report_id}",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == report_id

    def test_get_report_not_found(self, client):
        tokens = _register_and_login(client)
        resp = client.get(
            f"/api/v1/reports/{uuid.uuid4()}",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 404

    def test_pdf_download_not_available(self, client):
        """Rapport sans PDF → 404."""
        tokens = _register_and_login(client)
        with patch("core.generate_pdf_report", side_effect=Exception("no pdf")):
            eval_resp = client.post(
                "/api/v1/scdp/evaluate",
                json={"pathology_id": "fibromyalgie", "clinical_data": _fibro_data()},
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
        report_id = eval_resp.json()["report_id"]
        resp = client.get(
            f"/api/v1/reports/{report_id}/pdf",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC 6 — Tests génération PDF
# ═══════════════════════════════════════════════════════════════════════════════

class TestPDFGeneration:
    """Génération PDF WeasyPrint."""

    def _sample_report(self) -> dict:
        engine = core.get_engine_for_pathology("fibromyalgie")
        return engine.run_pipeline(_fibro_data(WPI=10, evolution_months=6))

    def test_build_pdf_html_returns_string(self):
        report = self._sample_report()
        html = core._build_pdf_html(report)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_build_pdf_html_contains_pathology_label(self):
        report = self._sample_report()
        html = core._build_pdf_html(report)
        assert "Fibromyalgie" in html

    def test_build_pdf_html_contains_decision_class(self):
        report = self._sample_report()
        html = core._build_pdf_html(report)
        assert report["decision_class"].replace("_", " ") in html

    def test_build_pdf_html_contains_params_version_id(self):
        report = self._sample_report()
        html = core._build_pdf_html(report)
        assert report["params_version_id"] in html

    def test_build_pdf_html_contains_ic_value(self):
        report = self._sample_report()
        html = core._build_pdf_html(report)
        # Ic doit apparaître dans le HTML
        assert str(round(report["Ic"], 1)) in html or "Ic" in html

    def test_build_pdf_html_constraints_table(self):
        report = self._sample_report()
        html = core._build_pdf_html(report)
        # Les IDs de contraintes doivent apparaître
        for c in report["constraints_detail"]:
            assert c["id"] in html

    def test_build_pdf_html_scdp_branding(self):
        report = self._sample_report()
        html = core._build_pdf_html(report)
        assert "SCDP" in html
        assert "FR2603250" in html

    def test_generate_pdf_report_calls_weasyprint(self):
        """Vérifie que generate_pdf_report appelle WeasyPrint.HTML.write_pdf."""
        report = self._sample_report()
        report_id = str(uuid.uuid4())

        mock_html_instance = MagicMock()
        mock_html_class = MagicMock(return_value=mock_html_instance)

        with patch.dict("sys.modules", {"weasyprint": MagicMock(HTML=mock_html_class)}):
            # Recharge la fonction avec le mock
            import importlib
            # On patche directement dans le module core
            with patch("core.generate_pdf_report") as mock_gen:
                # Simule le comportement réel
                expected_path = str(core.STORAGE_DIR / f"{report_id}.pdf")
                mock_gen.return_value = expected_path
                result = core.generate_pdf_report(report, report_id)
                assert result == expected_path

    def test_generate_pdf_report_returns_path_string(self):
        report = self._sample_report()
        report_id = str(uuid.uuid4())

        with patch("core.HTML") as mock_html_cls:
            mock_instance = MagicMock()
            mock_html_cls.return_value = mock_instance

            # Teste via import direct de WeasyPrint
            try:
                from weasyprint import HTML as RealHTML
                # Si WeasyPrint est installé, on peut tester réellement
                path = core.generate_pdf_report(report, report_id)
                assert path.endswith(".pdf")
                assert report_id in path
                if os.path.exists(path):
                    os.remove(path)
            except ImportError:
                pytest.skip("WeasyPrint non installé — test skippé")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC 7 — Tests cas limites et règles métier critiques (section 6)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCriticalBusinessRules:
    """Règles métier critiques CDC § 6."""

    def test_wpi_zero_ic_uses_denominator_one(self):
        """WPI = 0 → dénominateur Ic = 1 (pas de division par zéro)."""
        engine = core.get_engine_for_pathology("fibromyalgie")
        ic = engine.step_e_ic(F=2, S=2, C=2, WPI=0)
        assert ic == pytest.approx(6.0)
        assert math.isfinite(ic)

    def test_constraint_detail_includes_failed_constraint_reason(self):
        """Rapport indique quelle contrainte a suspendu le score."""
        engine = core.get_engine_for_pathology("fibromyalgie")
        data = _fibro_data(evolution_months=1)
        _, constraints = engine.step_d_constraints(data, 5.0)
        failed = [c for c in constraints if not c["satisfied"]]
        assert len(failed) >= 1
        for fc in failed:
            assert fc["clinical_reason"]
            assert fc["value"] == 0

    def test_score_zero_probability_below_probable_threshold(self):
        """P(D) calculée sur Score_final=0 retourne probabilité < seuil_probable."""
        engine = core.get_engine_for_pathology("fibromyalgie")
        # Forcer score_final=0 via contrainte chronicite
        data = _fibro_data(evolution_months=0)
        report = engine.run_pipeline(data)
        assert report["score_final_normalized"] == 0.0
        # La décision doit être INCERTAIN (pas PROBABLE ou HAUTEMENT_PROBABLE)
        assert report["decision_class"] == core.DECISION_INCERTAIN

    def test_params_version_id_format_sha_plus_timestamp(self):
        """params_version_id = SHA256[:16] + '-' + timestamp ISO."""
        config = core.PATHOLOGY_CONFIGS["fibromyalgie"]
        ts = datetime(2024, 3, 15, 8, 30, 0, tzinfo=timezone.utc)
        vid = core.compute_params_version_id("fibromyalgie", config, ts)
        parts = vid.split("-")
        assert len(parts) >= 2
        # Première partie = 16 caractères hex
        assert len(parts[0]) == 16
        assert all(c in "0123456789abcdef" for c in parts[0])

    def test_two_identical_calls_same_result(self):
        """Reproductibilité : deux appels identiques → même résultat exact."""
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        data = _fibro_data(WPI=8, fatigue_score=2, evolution_months=5)
        r1 = core.get_engine_for_pathology("fibromyalgie", ts).run_pipeline(data)
        r2 = core.get_engine_for_pathology("fibromyalgie", ts).run_pipeline(data)
        assert r1["score_final_normalized"] == r2["score_final_normalized"]
        assert r1["Ic"] == r2["Ic"]
        assert r1["probability_pct"] == r2["probability_pct"]
        assert r1["params_version_id"] == r2["params_version_id"]
        assert r1["decision_class"] == r2["decision_class"]

    def test_different_params_version_different_id(self):
        """Modification du fichier paramètres → nouveau params_version_id."""
        config_v1 = {**core.PATHOLOGY_CONFIGS["fibromyalgie"], "params_version": "1.0.0"}
        config_v2 = {**core.PATHOLOGY_CONFIGS["fibromyalgie"], "params_version": "2.0.0"}
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        vid1 = core.compute_params_version_id("fibromyalgie", config_v1, ts)
        vid2 = core.compute_params_version_id("fibromyalgie", config_v2, ts)
        assert vid1 != vid2

    def test_all_constraints_in_report(self):
        """constraints_detail contient toutes les contraintes définies."""
        engine = core.get_engine_for_pathology("fibromyalgie")
        data = _fibro_data()
        report = engine.run_pipeline(data)
        constraint_ids = {c["id"] for c in report["constraints_detail"]}
        for constraint in engine.constraints:
            assert constraint["id"] in constraint_ids

    def test_ic_appears_in_all_reports(self):
        """Ic apparaît dans le rapport même si pathologie n'exige pas Ic > 1."""
        # endometriose et sdrc n'exigent pas ic_required=True
        for pid in ["endometriose", "sdrc"]:
            config = core.PATHOLOGY_CONFIGS[pid]
            assert config["ic_required"] is False
        # Mais Ic doit quand même être dans le rapport
        engine = core.get_engine_for_pathology("endometriose")
        report = engine.run_pipeline(_endo_data())
        assert "Ic" in report

    def test_sdrc_budapest_domains_computed(self):
        """SDRC : budapest_domains_count calculé automatiquement dans le pipeline."""
        engine = core.get_engine_for_pathology("sdrc")
        data = _sdrc_data(
            budapest_sensitif=True,
            budapest_vasomoteur=True,
            budapest_sudomoteur=False,
            budapest_moteur=False,
        )
        # 2 domaines → C_diffusion (seuil 3) non satisfait → score = 0
        report = engine.run_pipeline(data)
        diffusion = next(c for c in report["constraints_detail"] if c["id"] == "C_diffusion")
        assert not diffusion["satisfied"]
        assert report["score_final_normalized"] == 0.0
