"""
SCDP — Moteur algorithmique, logique métier, PDF, hashing, métriques, logs
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary

# ─────────────────────────────────────────────
# Logging structuré
# ─────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger("scdp")


# ─────────────────────────────────────────────
# Métriques Prometheus
# ─────────────────────────────────────────────

SCDP_EVALUATIONS_TOTAL = Counter(
    "scdp_evaluations_total",
    "Nombre total d'évaluations SCDP",
    ["pathology_id", "decision_class"],
)
SCDP_EVALUATION_DURATION = Histogram(
    "scdp_evaluation_duration_seconds",
    "Durée du pipeline SCDP en secondes",
    ["pathology_id"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
SCDP_ERRORS_TOTAL = Counter(
    "scdp_errors_total",
    "Nombre d'erreurs pipeline SCDP",
    ["error_type"],
)
SCDP_BATCH_SIZE = Histogram(
    "scdp_batch_size",
    "Taille des lots d'évaluation",
    buckets=[1, 5, 10, 25, 50, 100],
)
PDF_GENERATION_DURATION = Histogram(
    "scdp_pdf_generation_duration_seconds",
    "Durée génération PDF",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)
ACTIVE_USERS = Gauge("scdp_active_users_total", "Utilisateurs actifs enregistrés")


# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────

STORAGE_DIR = Path("storage/reports")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DECISION_INCERTAIN = "INCERTAIN"
DECISION_PROBABLE = "PROBABLE"
DECISION_HAUTEMENT_PROBABLE = "HAUTEMENT_PROBABLE"


# ─────────────────────────────────────────────
# Paramètres des 5 pathologies (données internes)
# ─────────────────────────────────────────────

PATHOLOGY_CONFIGS: dict[str, dict] = {
    "fibromyalgie": {
        "pathology_id": "fibromyalgie",
        "pathology_label": "Fibromyalgie",
        "params_version": "1.0.0",
        "reference_standard": "Critères ACR 2016",
        "weights": {
            "alpha": 1.0,   # WPI normalisé
            "beta": 2.0,    # fatigue_score normalisé
            "gamma": 1.5,   # sleep_score normalisé
            "delta": 1.0,   # cognitive_score normalisé
            "epsilon": 0.5, # psych_score normalisé
        },
        "penalization_weights": {
            "v1": 3.0,  # exclusion flag 1 — pathologie organique explicative
            "v2": 2.5,  # exclusion flag 2 — inflammation systémique
            "v3": 2.0,  # exclusion flag 3 — autre
        },
        "constraints": [
            {
                "id": "C_chronicite",
                "label": "Chronicité ≥ 3 mois",
                "type": "chronicite",
                "threshold": 3,
                "field": "evolution_months",
                "operator": "gte",
                "clinical_reason": "La fibromyalgie requiert une évolution symptomatique d'au moins 3 mois (ACR 2016).",
            },
            {
                "id": "C_diffusion",
                "label": "Diffusion douloureuse WPI ≥ 7",
                "type": "diffusion",
                "threshold": 7,
                "field": "WPI",
                "operator": "gte",
                "clinical_reason": "Une atteinte diffuse (WPI ≥ 7) est requise selon les critères ACR 2016.",
            },
            {
                "id": "C_exclusion",
                "label": "Absence de pathologie organique explicative",
                "type": "exclusion",
                "field": "exclusion_flags",
                "operator": "none_true",
                "clinical_reason": "La présence d'une pathologie organique explicative exclut le diagnostic de fibromyalgie.",
            },
        ],
        "logistic_params": {
            "a": -3.5,
            "b_vector": [0.8, 1.6, 1.2, 0.8, 0.4],
            "c": 1.2,
        },
        "decision_thresholds": {
            "probable": 18.0,
            "haute": 30.0,
        },
        "ic_required": True,
        "ic_threshold": 1.0,
        "pathology_specific_fields": [],
        "score_max_theoretical": 6.0,  # Σ(wi × 1) = 1+2+1.5+1+0.5
    },

    "endometriose": {
        "pathology_id": "endometriose",
        "pathology_label": "Endométriose",
        "params_version": "1.0.0",
        "reference_standard": "Classification rASRM",
        "weights": {
            "alpha": 1.5,
            "beta": 1.5,
            "gamma": 1.0,
            "delta": 0.8,
            "epsilon": 1.0,
        },
        "penalization_weights": {
            "v1": 3.0,
            "v2": 2.0,
            "v3": 1.5,
        },
        "constraints": [
            {
                "id": "C_chronicite",
                "label": "Chronicité ≥ 6 mois",
                "type": "chronicite",
                "threshold": 6,
                "field": "evolution_months",
                "operator": "gte",
                "clinical_reason": "L'endométriose requiert une évolution d'au moins 6 mois pour classification (rASRM).",
            },
            {
                "id": "C_diffusion",
                "label": "Atteinte pelvienne multi-localisée (≥ 2 sites)",
                "type": "diffusion",
                "threshold": 2,
                "field": "pelvic_locations_count",
                "operator": "gte",
                "clinical_reason": "L'atteinte multi-localisée pelvienne (≥ 2 sites) est requise pour la classification rASRM.",
            },
            {
                "id": "C_exclusion",
                "label": "Absence de pathologie organique explicative alternative",
                "type": "exclusion",
                "field": "exclusion_flags",
                "operator": "none_true",
                "clinical_reason": "Une pathologie organique alternative explicative exclut le diagnostic d'endométriose.",
            },
        ],
        "logistic_params": {
            "a": -3.0,
            "b_vector": [1.2, 1.2, 0.8, 0.6, 0.8],
            "c": 0.8,
        },
        "decision_thresholds": {
            "probable": 16.0,
            "haute": 28.0,
        },
        "ic_required": False,
        "ic_threshold": 1.0,
        "pathology_specific_fields": [
            {
                "name": "cyclicite_score",
                "type": "integer",
                "range": [0, 3],
                "label": "Score de cyclicité des douleurs (0-3)",
                "required": True,
            },
            {
                "name": "dyspareunia_score",
                "type": "integer",
                "range": [0, 3],
                "label": "Score de dyspareunie (0-3)",
                "required": True,
            },
            {
                "name": "pelvic_locations_count",
                "type": "integer",
                "range": [0, 10],
                "label": "Nombre de sites pelviens atteints",
                "required": True,
            },
        ],
        "score_max_theoretical": 5.8,  # Σwi × 1 = 1.5+1.5+1+0.8+1
    },

    "sdrc": {
        "pathology_id": "sdrc",
        "pathology_label": "Syndrome Douloureux Régional Complexe (SDRC)",
        "params_version": "1.0.0",
        "reference_standard": "Critères de Budapest",
        "weights": {
            "alpha": 2.0,
            "beta": 1.0,
            "gamma": 1.0,
            "delta": 0.8,
            "epsilon": 0.5,
        },
        "penalization_weights": {
            "v1": 4.0,
            "v2": 3.0,
            "v3": 2.0,
        },
        "constraints": [
            {
                "id": "C_chronicite",
                "label": "Chronicité ≥ 1 mois post-événement déclenchant",
                "type": "chronicite",
                "threshold": 1,
                "field": "evolution_months",
                "operator": "gte",
                "clinical_reason": "Le SDRC doit persister au moins 1 mois après l'événement déclenchant (Budapest).",
            },
            {
                "id": "C_diffusion",
                "label": "Atteinte ≥ 3 domaines Budapest",
                "type": "diffusion",
                "threshold": 3,
                "field": "budapest_domains_count",
                "operator": "gte",
                "clinical_reason": "Les critères de Budapest exigent la présence d'au moins 3 domaines sur 4.",
            },
            {
                "id": "C_exclusion",
                "label": "Absence de pathologie organique explicative",
                "type": "exclusion",
                "field": "exclusion_flags",
                "operator": "none_true",
                "clinical_reason": "Une pathologie organique explicative des symptômes exclut le SDRC.",
            },
            {
                "id": "C_trigger_event",
                "label": "Événement déclenchant identifié",
                "type": "trigger_event",
                "field": "trigger_event",
                "operator": "is_true",
                "clinical_reason": "Le SDRC requiert un événement déclenchant identifiable (Budapest).",
            },
        ],
        "logistic_params": {
            "a": -3.2,
            "b_vector": [1.6, 0.8, 0.8, 0.64, 0.4],
            "c": 0.6,
        },
        "decision_thresholds": {
            "probable": 14.0,
            "haute": 25.0,
        },
        "ic_required": False,
        "ic_threshold": 1.0,
        "pathology_specific_fields": [
            {
                "name": "trigger_event",
                "type": "boolean",
                "label": "Événement déclenchant identifié (traumatisme, chirurgie, etc.)",
                "required": True,
            },
            {
                "name": "budapest_sensitif",
                "type": "boolean",
                "label": "Domaine sensitif Budapest (hyperalgésie/allodynie)",
                "required": True,
            },
            {
                "name": "budapest_vasomoteur",
                "type": "boolean",
                "label": "Domaine vasomoteur Budapest (asymétrie température/couleur peau)",
                "required": True,
            },
            {
                "name": "budapest_sudomoteur",
                "type": "boolean",
                "label": "Domaine sudomoteur/oedème Budapest",
                "required": True,
            },
            {
                "name": "budapest_moteur",
                "type": "boolean",
                "label": "Domaine moteur/trophique Budapest",
                "required": True,
            },
        ],
        "score_max_theoretical": 5.3,  # Σwi × 1 = 2+1+1+0.8+0.5
    },

    "sfc_me": {
        "pathology_id": "sfc_me",
        "pathology_label": "Syndrome de Fatigue Chronique / Encéphalomyélite Myalgique (SFC/ME)",
        "params_version": "1.0.0",
        "reference_standard": "Critères IOM 2015",
        "weights": {
            "alpha": 0.8,
            "beta": 2.5,
            "gamma": 1.2,
            "delta": 1.5,
            "epsilon": 0.8,
        },
        "penalization_weights": {
            "v1": 3.0,
            "v2": 2.5,
            "v3": 2.0,
        },
        "constraints": [
            {
                "id": "C_chronicite",
                "label": "Chronicité ≥ 6 mois",
                "type": "chronicite",
                "threshold": 6,
                "field": "evolution_months",
                "operator": "gte",
                "clinical_reason": "Le SFC/ME requiert une durée d'au moins 6 mois (critères IOM 2015).",
            },
            {
                "id": "C_diffusion",
                "label": "Fatigue sévère et invalidante (score ≥ 2)",
                "type": "diffusion",
                "threshold": 2,
                "field": "fatigue_score",
                "operator": "gte",
                "clinical_reason": "Une fatigue sévère et invalidante est le critère central du SFC/ME (IOM 2015).",
            },
            {
                "id": "C_exclusion",
                "label": "Absence de pathologie médicale ou psychiatrique explicative",
                "type": "exclusion",
                "field": "exclusion_flags",
                "operator": "none_true",
                "clinical_reason": "Une cause médicale ou psychiatrique explicative exclut le diagnostic de SFC/ME.",
            },
        ],
        "logistic_params": {
            "a": -3.8,
            "b_vector": [0.64, 2.0, 0.96, 1.2, 0.64],
            "c": 1.5,
        },
        "decision_thresholds": {
            "probable": 17.0,
            "haute": 29.0,
        },
        "ic_required": True,
        "ic_threshold": 1.0,
        "pathology_specific_fields": [],
        "score_max_theoretical": 6.8,  # Σwi × 1 = 0.8+2.5+1.2+1.5+0.8
    },

    "covid_long_neurologique": {
        "pathology_id": "covid_long_neurologique",
        "pathology_label": "Covid Long Neurologique",
        "params_version": "1.0.0",
        "reference_standard": "Critères OMS Covid Long 2021",
        "weights": {
            "alpha": 0.8,
            "beta": 2.5,
            "gamma": 1.5,
            "delta": 2.0,
            "epsilon": 1.0,
        },
        "penalization_weights": {
            "v1": 3.5,
            "v2": 2.5,
            "v3": 2.0,
        },
        "constraints": [
            {
                "id": "C_chronicite",
                "label": "Chronicité ≥ 3 mois post-infection",
                "type": "chronicite",
                "threshold": 3,
                "field": "evolution_months",
                "operator": "gte",
                "clinical_reason": "Le Covid Long est défini par des symptômes persistants ≥ 3 mois post-infection (OMS 2021).",
            },
            {
                "id": "C_diffusion",
                "label": "Infection Covid-19 confirmée",
                "type": "diffusion",
                "field": "covid_confirmed",
                "operator": "is_true",
                "clinical_reason": "Une infection Covid-19 confirmée est requise pour le diagnostic de Covid Long.",
            },
            {
                "id": "C_exclusion",
                "label": "Absence de cause neurologique alternative",
                "type": "exclusion",
                "field": "exclusion_flags",
                "operator": "none_true",
                "clinical_reason": "Une cause neurologique alternative explicative exclut le Covid Long Neurologique.",
            },
        ],
        "logistic_params": {
            "a": -3.3,
            "b_vector": [0.64, 2.0, 1.2, 1.6, 0.8],
            "c": 1.3,
        },
        "decision_thresholds": {
            "probable": 16.0,
            "haute": 28.0,
        },
        "ic_required": True,
        "ic_threshold": 1.0,
        "pathology_specific_fields": [
            {
                "name": "covid_confirmed",
                "type": "boolean",
                "label": "Infection Covid-19 confirmée (PCR/sérologie/clinique)",
                "required": True,
            },
            {
                "name": "onset_date",
                "type": "date",
                "label": "Date de l'infection initiale (YYYY-MM-DD)",
                "required": True,
            },
        ],
        "score_max_theoretical": 7.8,  # Σwi × 1 = 0.8+2.5+1.5+2+1
    },
}


# ─────────────────────────────────────────────
# Hashing SHA256
# ─────────────────────────────────────────────

def compute_sha256(data: dict | str) -> str:
    """Calcule le hash SHA256 d'un dictionnaire ou d'une chaîne."""
    if isinstance(data, dict):
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
    else:
        payload = data
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_params_version_id(pathology_id: str, params: dict, loaded_at: datetime) -> str:
    """Génère le params_version_id : SHA256(params) + timestamp ISO."""
    sha = compute_sha256(params)
    ts = loaded_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{sha[:16]}-{ts}"


def compute_refresh_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────
# Validation des entrées cliniques
# ─────────────────────────────────────────────

class SCDPValidationError(Exception):
    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__(str(errors))


def validate_clinical_data(
    pathology_id: str,
    clinical_data: dict,
    config: dict,
) -> list[dict]:
    """
    Valide les données cliniques selon les règles SCDP.
    Retourne une liste d'erreurs (vide si OK).
    """
    errors: list[dict] = []

    # WPI
    wpi = clinical_data.get("WPI")
    if wpi is None:
        errors.append({"field": "WPI", "msg": "Le champ WPI est obligatoire."})
    elif not isinstance(wpi, int) or wpi < 0 or wpi > 19:
        errors.append({"field": "WPI", "msg": "WPI doit être un entier entre 0 et 19 inclus."})

    # Scores 0-3
    for score_field in ["fatigue_score", "sleep_score", "cognitive_score", "psych_score"]:
        val = clinical_data.get(score_field)
        if val is None:
            errors.append({"field": score_field, "msg": f"Le champ {score_field} est obligatoire."})
        elif not isinstance(val, int) or val < 0 or val > 3:
            errors.append({"field": score_field, "msg": f"{score_field} doit être un entier entre 0 et 3 inclus."})

    # evolution_months
    evo = clinical_data.get("evolution_months")
    if evo is None:
        errors.append({"field": "evolution_months", "msg": "Le champ evolution_months est obligatoire."})
    elif not isinstance(evo, int) or evo < 0:
        errors.append({"field": "evolution_months", "msg": "evolution_months doit être un entier ≥ 0."})

    # exclusion_flags
    flags = clinical_data.get("exclusion_flags")
    if flags is None:
        errors.append({"field": "exclusion_flags", "msg": "Le champ exclusion_flags est obligatoire (tableau de booléens)."})
    elif not isinstance(flags, list) or not all(isinstance(f, bool) for f in flags):
        errors.append({"field": "exclusion_flags", "msg": "exclusion_flags doit être un tableau de booléens."})

    # Champs pathologie-spécifiques
    specific_fields = config.get("pathology_specific_fields", [])
    missing_specific: list[str] = []
    for field_def in specific_fields:
        fname = field_def["name"]
        ftype = field_def["type"]
        required = field_def.get("required", True)

        val = clinical_data.get(fname)
        if val is None:
            if required:
                missing_specific.append(fname)
            continue

        # Type checking
        if ftype == "integer":
            if not isinstance(val, int):
                errors.append({"field": fname, "msg": f"{fname} doit être un entier."})
            else:
                frange = field_def.get("range")
                if frange and (val < frange[0] or val > frange[1]):
                    errors.append({
                        "field": fname,
                        "msg": f"{fname} doit être entre {frange[0]} et {frange[1]} inclus.",
                    })
        elif ftype == "boolean":
            if not isinstance(val, bool):
                errors.append({"field": fname, "msg": f"{fname} doit être un booléen."})
        elif ftype == "date":
            if not isinstance(val, str):
                errors.append({"field": fname, "msg": f"{fname} doit être une chaîne date ISO (YYYY-MM-DD)."})
            else:
                try:
                    parsed_date = date.fromisoformat(val)
                    if parsed_date > date.today():
                        errors.append({"field": fname, "msg": f"{fname} ne peut pas être dans le futur."})
                    if parsed_date < date(2020, 1, 1):
                        errors.append({"field": fname, "msg": f"{fname} doit être postérieure au 01/01/2020."})
                except ValueError:
                    errors.append({"field": fname, "msg": f"{fname} format invalide. Attendu: YYYY-MM-DD."})

    if missing_specific:
        errors.append({
            "field": "pathology_specific_fields",
            "msg": f"Champs manquants pour la pathologie '{pathology_id}': {', '.join(missing_specific)}",
            "missing_fields": missing_specific,
        })

    return errors


# ─────────────────────────────────────────────
# Pipeline SCDP — 7 étapes
# ─────────────────────────────────────────────

class SCDPEngine:
    """
    Moteur SCDP — implémentation stricte des 7 étapes du brevet FR2603250.
    """

    def __init__(self, config: dict, params_version_id: str):
        self.config = config
        self.params_version_id = params_version_id
        self.weights = config["weights"]
        self.penalization_weights = config["penalization_weights"]
        self.constraints = config["constraints"]
        self.logistic_params = config["logistic_params"]
        self.thresholds = config["decision_thresholds"]
        self.score_max = config.get("score_max_theoretical", 6.0)

    # ──── Étape b — Transformation / Normalisation ────

    def _normalize_wpi(self, wpi: int) -> float:
        """Normalisation min-max WPI : [0-19] → [0,1]"""
        return wpi / 19.0

    def _normalize_score_0_3(self, val: int) -> float:
        """Normalisation min-max scores 0-3 → [0,1]"""
        return val / 3.0

    def _normalize_specific_field(self, field_def: dict, val: Any) -> float:
        """Normalise un champ pathologie-spécifique numérique."""
        if field_def["type"] == "integer":
            frange = field_def.get("range", [0, 1])
            span = frange[1] - frange[0]
            if span == 0:
                return 0.0
            return (val - frange[0]) / span
        elif field_def["type"] == "boolean":
            return 1.0 if val else 0.0
        return 0.0

    def step_b_transform(self, clinical_data: dict) -> dict:
        """
        Étape b — Transformation en variables numériques normalisées.
        Retourne Xi (variables positives) et Ej (variables d'exclusion).
        """
        Xi = [
            self._normalize_wpi(clinical_data["WPI"]),
            self._normalize_score_0_3(clinical_data["fatigue_score"]),
            self._normalize_score_0_3(clinical_data["sleep_score"]),
            self._normalize_score_0_3(clinical_data["cognitive_score"]),
            self._normalize_score_0_3(clinical_data["psych_score"]),
        ]

        # Variables d'exclusion Ej (exclusion_flags)
        flags = clinical_data.get("exclusion_flags", [])
        Ej = [1.0 if f else 0.0 for f in flags]

        # Pad Ej à la longueur des penalization_weights si nécessaire
        pv_keys = list(self.penalization_weights.values())
        while len(Ej) < len(pv_keys):
            Ej.append(0.0)
        Ej = Ej[: len(pv_keys)]

        return {"Xi": Xi, "Ej": Ej}

    # ──── Étape c — Score brut ────

    def step_c_score_brut(self, Xi: list[float], Ej: list[float]) -> float:
        """
        Étape c — Score brut pondéré.
        Score_brut = Σ(wi × Xi) − Σ(vj × Ej)
        """
        wi_values = list(self.weights.values())
        score_positif = sum(wi_values[i] * Xi[i] for i in range(min(len(wi_values), len(Xi))))
        vj_values = list(self.penalization_weights.values())
        score_negatif = sum(vj_values[j] * Ej[j] for j in range(min(len(vj_values), len(Ej))))
        return score_positif - score_negatif

    # ──── Étape d — Contraintes binaires ────

    def _evaluate_constraint(self, constraint: dict, clinical_data: dict) -> tuple[int, str]:
        """
        Évalue une contrainte Ck.
        Retourne (0 ou 1, raison clinique).
        """
        op = constraint["operator"]
        field = constraint["field"]
        threshold = constraint.get("threshold")

        if op == "gte":
            # Récupérer la valeur du champ (commun ou spécifique)
            val = clinical_data.get(field)
            if val is None:
                return 0, f"Champ '{field}' absent — contrainte non satisfaite."
            satisfied = int(val >= threshold)
            reason = (
                constraint["clinical_reason"]
                if satisfied
                else f"{field} = {val} < seuil {threshold} — {constraint['clinical_reason']}"
            )
            return satisfied, reason

        elif op == "none_true":
            flags = clinical_data.get(field, [])
            any_true = any(flags)
            satisfied = 0 if any_true else 1
            reason = (
                constraint["clinical_reason"]
                if satisfied
                else f"Flag d'exclusion positif détecté — {constraint['clinical_reason']}"
            )
            return satisfied, reason

        elif op == "is_true":
            val = clinical_data.get(field)
            satisfied = 1 if val is True else 0
            reason = (
                constraint["clinical_reason"]
                if satisfied
                else f"'{field}' est absent ou False — {constraint['clinical_reason']}"
            )
            return satisfied, reason

        else:
            return 0, f"Opérateur inconnu '{op}' — contrainte non satisfaite par sécurité."

    def step_d_constraints(self, clinical_data: dict, score_brut: float) -> tuple[float, list[dict]]:
        """
        Étape d — Application des contraintes binaires.
        Score_final = Score_brut × C1 × C2 × ... × Cn
        Si un seul Ck = 0 → Score_final = 0
        """
        constraints_detail = []
        score_final = score_brut

        for constraint in self.constraints:
            ck, reason = self._evaluate_constraint(constraint, clinical_data)
            satisfied = ck == 1
            constraints_detail.append({
                "id": constraint["id"],
                "label": constraint["label"],
                "value": ck,
                "satisfied": satisfied,
                "clinical_reason": reason,
            })
            score_final *= ck  # Si ck=0 → score_final=0 immédiatement

        return score_final, constraints_detail

    # ──── Étape e — Indice Ic ────

    def step_e_ic(self, F: int, S: int, C: int, WPI: int) -> float:
        """
        Étape e — Indice de centralisation.
        Ic = (F + S + C) / (WPI + 1)
        WPI = 0 → dénominateur = 1 (unitarisation, pas de division par zéro)
        """
        return (F + S + C) / (WPI + 1)

    # ──── Étape f — Probabilité diagnostique ────

    def step_f_probability(self, Xi: list[float], Ic: float) -> float:
        """
        Étape f — Probabilité diagnostique.
        P(D) = 1 / (1 + e^-(a + Σ(bi × Xi) + c × Ic))
        """
        a = self.logistic_params["a"]
        b_vector = self.logistic_params["b_vector"]
        c = self.logistic_params["c"]

        linear_combination = a + sum(b_vector[i] * Xi[i] for i in range(min(len(b_vector), len(Xi)))) + c * Ic
        # Clamp pour éviter overflow float
        linear_combination = max(-500.0, min(500.0, linear_combination))
        probability = 1.0 / (1.0 + math.exp(-linear_combination))
        return probability

    # ──── Étape g — Rapport structuré ────

    def step_g_report(
        self,
        clinical_data: dict,
        score_final: float,
        Ic: float,
        probability: float,
        constraints_detail: list[dict],
        computed_at: datetime,
    ) -> dict:
        """
        Étape g — Construction du rapport structuré JSON.
        """
        # Normalisation score 0-100
        score_max = self.score_max if self.score_max > 0 else 1.0
        score_final_normalized = min(100.0, max(0.0, (score_final / score_max) * 100.0))

        # Probabilité en pourcentage
        probability_pct = round(probability * 100.0, 2)

        # Classification
        if score_final == 0.0:
            decision_class = DECISION_INCERTAIN
        elif score_final_normalized >= self.thresholds["haute"]:
            decision_class = DECISION_HAUTEMENT_PROBABLE
        elif score_final_normalized >= self.thresholds["probable"]:
            decision_class = DECISION_PROBABLE
        else:
            decision_class = DECISION_INCERTAIN

        # ic_signature
        ic_signature = "sensibilisation_centrale" if Ic > 1.0 else "profil_peripherique"

        return {
            "score_final_normalized": round(score_final_normalized, 4),
            "Ic": round(Ic, 4),
            "ic_signature": ic_signature,
            "probability_pct": probability_pct,
            "decision_class": decision_class,
            "constraints_detail": constraints_detail,
            "input_trace": {k: v for k, v in clinical_data.items()},
            "params_version_id": self.params_version_id,
            "pathology_id": self.config["pathology_id"],
            "pathology_label": self.config["pathology_label"],
            "computed_at": computed_at.isoformat(),
        }

    # ──── Pipeline complet ────

    def run_pipeline(self, clinical_data: dict) -> dict:
        """
        Exécute le pipeline complet SCDP étapes a→g.
        """
        computed_at = datetime.now(timezone.utc)

        # Étape b
        transformed = self.step_b_transform(clinical_data)
        Xi = transformed["Xi"]
        Ej = transformed["Ej"]

        # Étape c
        score_brut = self.step_c_score_brut(Xi, Ej)

        # SDRC — calcul budapest_domains_count à injecter dans clinical_data
        if self.config["pathology_id"] == "sdrc":
            domains = [
                clinical_data.get("budapest_sensitif", False),
                clinical_data.get("budapest_vasomoteur", False),
                clinical_data.get("budapest_sudomoteur", False),
                clinical_data.get("budapest_moteur", False),
            ]
            clinical_data = dict(clinical_data)  # copie pour ne pas muter l'original
            clinical_data["budapest_domains_count"] = sum(1 for d in domains if d is True)

        # Étape d
        score_final, constraints_detail = self.step_d_constraints(clinical_data, score_brut)

        # Étape e
        Ic = self.step_e_ic(
            clinical_data["fatigue_score"],
            clinical_data["sleep_score"],
            clinical_data["cognitive_score"],
            clinical_data["WPI"],
        )

        # Étape f
        probability = self.step_f_probability(Xi, Ic)

        # Étape g
        report = self.step_g_report(
            clinical_data=clinical_data,
            score_final=score_final,
            Ic=Ic,
            probability=probability,
            constraints_detail=constraints_detail,
            computed_at=computed_at,
        )

        logger.info(
            "scdp.pipeline.complete",
            pathology_id=self.config["pathology_id"],
            decision_class=report["decision_class"],
            score_final_normalized=report["score_final_normalized"],
            probability_pct=report["probability_pct"],
            Ic=report["Ic"],
        )

        return report


# ─────────────────────────────────────────────
# Factory moteur SCDP
# ─────────────────────────────────────────────

def get_engine_for_pathology(pathology_id: str, loaded_at: Optional[datetime] = None) -> SCDPEngine:
    """
    Retourne un moteur SCDP configuré pour la pathologie donnée.
    """
    config = PATHOLOGY_CONFIGS.get(pathology_id)
    if config is None:
        raise ValueError(f"Pathologie inconnue : '{pathology_id}'")

    if loaded_at is None:
        loaded_at = datetime.now(timezone.utc)

    params_version_id = compute_params_version_id(pathology_id, config, loaded_at)
    return SCDPEngine(config=config, params_version_id=params_version_id)


def get_pathology_config(pathology_id: str) -> Optional[dict]:
    return PATHOLOGY_CONFIGS.get(pathology_id)


def list_pathology_ids() -> list[str]:
    return list(PATHOLOGY_CONFIGS.keys())


def get_pathology_schema(pathology_id: str) -> Optional[dict]:
    """
    Retourne le JSON Schema des champs clinical_data pour une pathologie.
    """
    config = PATHOLOGY_CONFIGS.get(pathology_id)
    if not config:
        return None

    properties = {
        "WPI": {"type": "integer", "minimum": 0, "maximum": 19, "description": "Widespread Pain Index"},
        "fatigue_score": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Score de fatigue"},
        "sleep_score": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Score de sommeil"},
        "cognitive_score": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Score cognitif"},
        "psych_score": {"type": "integer", "minimum": 0, "maximum": 3, "description": "Score psychologique"},
        "evolution_months": {"type": "integer", "minimum": 0, "description": "Durée évolution en mois"},
        "exclusion_flags": {
            "type": "array",
            "items": {"type": "boolean"},
            "description": "Flags d'exclusion — biomarqueurs organiques",
        },
    }
    required = ["WPI", "fatigue_score", "sleep_score", "cognitive_score", "psych_score", "evolution_months", "exclusion_flags"]

    for field_def in config.get("pathology_specific_fields", []):
        fname = field_def["name"]
        ftype = field_def["type"]
        label = field_def.get("label", fname)

        if ftype == "integer":
            frange = field_def.get("range", [0, 100])
            properties[fname] = {
                "type": "integer",
                "minimum": frange[0],
                "maximum": frange[1],
                "description": label,
            }
        elif ftype == "boolean":
            properties[fname] = {"type": "boolean", "description": label}
        elif ftype == "date":
            properties[fname] = {
                "type": "string",
                "format": "date",
                "description": label,
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
            }

        if field_def.get("required", True):
            required.append(fname)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "title": f"Schéma clinical_data — {config['pathology_label']}",
        "required": required,
        "properties": properties,
        "additionalProperties": False,
    }


# ─────────────────────────────────────────────
# Génération PDF
# ─────────────────────────────────────────────

def generate_pdf_report(report: dict, report_id: str) -> str:
    """
    Génère un rapport PDF complet pour une évaluation SCDP.
    Retourne le chemin absolu du fichier PDF généré.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    pdf_filename = f"{report_id}.pdf"
    pdf_path = STORAGE_DIR / pdf_filename

    pathology_label = report.get("pathology_label", report.get("pathology_id", "N/A"))
    computed_at = report.get("computed_at", "N/A")
    decision_class = report.get("decision_class", "INCERTAIN")
    score = report.get("score_final_normalized", 0.0)
    Ic = report.get("Ic", 0.0)
    ic_sig = report.get("ic_signature", "N/A").replace("_", " ")
    probability_pct = report.get("probability_pct", 0.0)
    params_version_id = report.get("params_version_id", "N/A")
    constraints = report.get("constraints_detail", [])
    input_trace = report.get("input_trace", {})

    # Couleurs par décision
    _decision_colors = {
        "HAUTEMENT_PROBABLE": colors.HexColor("#0f5132"),
        "PROBABLE": colors.HexColor("#664d03"),
        "INCERTAIN": colors.HexColor("#842029"),
    }
    _decision_bg_colors = {
        "HAUTEMENT_PROBABLE": colors.HexColor("#d1e7dd"),
        "PROBABLE": colors.HexColor("#fff3cd"),
        "INCERTAIN": colors.HexColor("#f8d7da"),
    }
    decision_fg = _decision_colors.get(decision_class, colors.HexColor("#495057"))
    decision_bg = _decision_bg_colors.get(decision_class, colors.HexColor("#e9ecef"))

    BLUE = colors.HexColor("#0d6efd")
    GREY = colors.HexColor("#6c757d")
    LIGHT_GREY_BG = colors.HexColor("#f8f9fa")
    BORDER = colors.HexColor("#dee2e6")
    TEXT_DARK = colors.HexColor("#1a1a2e")
    WARN_BG = colors.HexColor("#fff3cd")
    WARN_FG = colors.HexColor("#664d03")
    GREEN = colors.HexColor("#0f5132")
    RED = colors.HexColor("#842029")

    styles = getSampleStyleSheet()

    style_normal = ParagraphStyle(
        "SCDPNormal",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
    )
    style_small_grey = ParagraphStyle(
        "SCDPSmallGrey",
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=GREY,
    )
    style_section = ParagraphStyle(
        "SCDPSection",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=12,
        textColor=GREY,
        spaceBefore=14,
        spaceAfter=6,
    )
    style_brand = ParagraphStyle(
        "SCDPBrand",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=BLUE,
    )
    style_sub = ParagraphStyle(
        "SCDPSub",
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=GREY,
    )
    style_decision = ParagraphStyle(
        "SCDPDecision",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=decision_fg,
    )
    style_score_label = ParagraphStyle(
        "SCDPScoreLabel",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=GREY,
    )
    style_score_value = ParagraphStyle(
        "SCDPScoreValue",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=BLUE,
    )
    style_score_unit = ParagraphStyle(
        "SCDPScoreUnit",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=GREY,
    )
    style_warn = ParagraphStyle(
        "SCDPWarn",
        fontName="Helvetica",
        fontSize=8,
        leading=12,
        textColor=WARN_FG,
    )
    style_footer = ParagraphStyle(
        "SCDPFooter",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=GREY,
    )
    style_constraint_reason = ParagraphStyle(
        "SCDPReason",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=GREY,
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    story = []

    # ── En-tête ──
    header_data = [
        [
            [
                Paragraph("SCDP", style_brand),
                Paragraph("Système de Classification Diagnostique Probabiliste", style_sub),
                Spacer(1, 4),
                Paragraph("Brevet FR2603250 — INPI (dossier en cours)", style_small_grey),
            ],
            [
                Paragraph(f"<b>{pathology_label}</b>", style_normal),
                Paragraph(f"Calculé le : {computed_at}", style_small_grey),
                Paragraph(f"Version paramètres :", style_small_grey),
                Paragraph(f"<font size='7'>{params_version_id}</font>", style_small_grey),
            ],
        ]
    ]
    header_table = Table(header_data, colWidths=["60%", "40%"])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LINEBELOW", (0, 0), (-1, -1), 2, BLUE),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # ── Décision clinique ──
    story.append(Paragraph("DÉCISION CLINIQUE", style_section))
    decision_label = decision_class.replace("_", " ")
    decision_cell = Table(
        [[Paragraph(decision_label, style_decision)]],
        colWidths=["100%"],
    )
    decision_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), decision_bg),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ]))
    story.append(decision_cell)
    story.append(Spacer(1, 10))

    # ── Scores clés ──
    story.append(Paragraph("SCORES CLÉS", style_section))
    scores_data = [[
        [
            Paragraph("SCORE FINAL NORMALISÉ", style_score_label),
            Paragraph(f"{score:.1f}", style_score_value),
            Paragraph("/ 100", style_score_unit),
        ],
        [
            Paragraph("PROBABILITÉ DIAGNOSTIQUE", style_score_label),
            Paragraph(f"{probability_pct:.1f}", style_score_value),
            Paragraph("%", style_score_unit),
        ],
        [
            Paragraph("INDICE DE CENTRALISATION Ic", style_score_label),
            Paragraph(f"{Ic:.3f}", style_score_value),
            Paragraph(ic_sig, style_score_unit),
        ],
    ]]
    col_w = (doc.width - 8) / 3
    scores_table = Table(scores_data, colWidths=[col_w, col_w, col_w], hAlign="LEFT")
    scores_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY_BG),
        ("BOX", (0, 0), (0, 0), 0.5, BORDER),
        ("BOX", (1, 0), (1, 0), 0.5, BORDER),
        ("BOX", (2, 0), (2, 0), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(scores_table)
    story.append(Spacer(1, 6))

    # ── Contraintes binaires ──
    story.append(Paragraph("CONTRAINTES BINAIRES", style_section))
    constraints_header = [
        Paragraph("<b>ID</b>", style_normal),
        Paragraph("<b>Contrainte</b>", style_normal),
        Paragraph("<b>Statut</b>", style_normal),
        Paragraph("<b>Motif clinique</b>", style_normal),
    ]
    constraints_rows_data = [constraints_header]
    for c in constraints:
        icon = "✓" if c["satisfied"] else "✗"
        icon_color = GREEN if c["satisfied"] else RED
        icon_style = ParagraphStyle(
            "icon",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=icon_color,
            alignment=1,
        )
        constraints_rows_data.append([
            Paragraph(f"<b>{c['id']}</b>", style_normal),
            Paragraph(c["label"], style_normal),
            Paragraph(icon, icon_style),
            Paragraph(c["clinical_reason"], style_constraint_reason),
        ])
    cw = doc.width
    c_table = Table(
        constraints_rows_data,
        colWidths=[cw * 0.14, cw * 0.28, cw * 0.08, cw * 0.50],
    )
    c_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GREY_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 1.5, BORDER),
        ("LINEBELOW", (0, 1), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
    ]))
    story.append(c_table)
    story.append(Spacer(1, 6))

    # ── Données d'entrée ──
    story.append(Paragraph("DONNÉES D'ENTRÉE (TRAÇABILITÉ)", style_section))
    input_header = [
        Paragraph("<b>Champ</b>", style_normal),
        Paragraph("<b>Valeur</b>", style_normal),
    ]
    input_rows_data = [input_header]
    for k, v in input_trace.items():
        if k == "exclusion_flags":
            continue
        input_rows_data.append([
            Paragraph(str(k), style_small_grey),
            Paragraph(f"<b>{v}</b>", style_normal),
        ])
    flags = input_trace.get("exclusion_flags", [])
    if flags:
        input_rows_data.append([
            Paragraph("exclusion_flags", style_small_grey),
            Paragraph(f"<b>[{', '.join(str(f) for f in flags)}]</b>", style_normal),
        ])
    i_table = Table(input_rows_data, colWidths=[doc.width * 0.35, doc.width * 0.65])
    i_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_GREY_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 1.5, BORDER),
        ("LINEBELOW", (0, 1), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(i_table)
    story.append(Spacer(1, 10))

    # ── Avertissement confidentiel ──
    warn_table = Table(
        [[Paragraph(
            "⚠️  Document confidentiel — Usage médical exclusif — NDA en vigueur — YSDA / FORMIDOC",
            style_warn,
        )]],
        colWidths=["100%"],
    )
    warn_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ffecb5")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(warn_table)
    story.append(Spacer(1, 8))

    # ── Pied de page ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 4))
    footer_data = [[
        Paragraph("SCDP v1.0 — FR2603250 (INPI)", style_footer),
        Paragraph("Généré automatiquement — non signé électroniquement", style_footer),
    ]]
    footer_table = Table(footer_data, colWidths=["50%", "50%"])
    footer_table.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(footer_table)

    doc.build(story)

    logger.info("scdp.pdf.generated", report_id=report_id, path=str(pdf_path))
    return str(pdf_path)


# ─────────────────────────────────────────────
# Helpers sécurité
# ─────────────────────────────────────────────

def sanitize_string(value: str, max_length: int = 500) -> str:
    """Nettoie une chaîne : supprime les caractères de contrôle, tronque."""
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    return value[:max_length].strip()


def safe_json_loads(raw: str) -> Optional[dict]:
    """Parse JSON en sécurité."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None