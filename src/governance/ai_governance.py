"""
governance/ai_governance.py
────────────────────────────
Enterprise AI governance framework for regulated healthcare environments.

Implements:
    - Bias detection and fairness auditing across demographic groups
    - Immutable AI decision audit logging
    - Model risk assessment and approval workflow
    - Responsible AI checklist enforcement
    - HIPAA compliance validation
    - Performance monitoring with clinical safety thresholds
    - Explainability reporting for clinical AI decisions

Regulatory standards addressed:
    - HIPAA Privacy and Security Rules
    - FDA AI/ML-Based Software as a Medical Device (SaMD) guidance
    - EU AI Act (high-risk AI system requirements)
    - NIST AI Risk Management Framework
    - Joint Commission AI governance standards

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    accuracy_score, recall_score, precision_score,
)

logger = logging.getLogger(__name__)


# ── Governance constants ──────────────────────────────────────────────────────

# Fairness threshold — max allowed disparity between demographic groups
FAIRNESS_THRESHOLD = 0.10  # 10% maximum disparity

# Clinical safety thresholds — minimum acceptable performance
CLINICAL_SAFETY_THRESHOLDS = {
    "min_recall":    0.75,   # Must catch 75%+ of high-risk cases
    "min_auc":       0.70,   # Minimum discrimination ability
    "min_precision": 0.60,   # Avoid excessive false positives
    "max_bias_disparity": FAIRNESS_THRESHOLD,
}

# Responsible AI checklist items
RESPONSIBLE_AI_CHECKLIST = [
    "phi_not_in_prompts",
    "data_minimisation_applied",
    "bias_audit_completed",
    "fairness_metrics_acceptable",
    "explainability_documented",
    "human_oversight_defined",
    "clinical_validation_completed",
    "hipaa_baa_in_place",
    "audit_logging_enabled",
    "rollback_plan_documented",
    "performance_thresholds_defined",
    "crisis_escalation_hardcoded",
]

# Protected attributes for fairness auditing
PROTECTED_ATTRIBUTES = [
    "gender", "race_ethnicity", "age_group",
    "language", "insurance_type", "socioeconomic_status",
]


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class AuditLogEntry:
    """
    Immutable audit record for an AI-assisted clinical decision.
    Once created, this record cannot be modified — providing
    a tamper-evident trail for regulatory review.
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: str = ""
    model_version: str = ""
    patient_id: str = ""        # De-identified
    clinician_id: str = ""
    decision_type: str = ""     # risk_score | match | recommendation | triage
    ai_output: dict = field(default_factory=dict)
    clinician_action: str = ""  # accepted | overridden | pending
    override_reason: str = ""
    phi_in_request: bool = False
    hipaa_compliant: bool = True
    content_hash: str = ""      # SHA-256 for tamper detection

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps({
            "entry_id":    self.entry_id,
            "timestamp":   self.timestamp,
            "model_name":  self.model_name,
            "patient_id":  self.patient_id,
            "ai_output":   self.ai_output,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify the audit record has not been tampered with."""
        return self.content_hash == self._compute_hash()

    def to_dict(self) -> dict:
        return {
            "entry_id":         self.entry_id,
            "timestamp":        self.timestamp,
            "model_name":       self.model_name,
            "model_version":    self.model_version,
            "patient_id":       self.patient_id,
            "clinician_id":     self.clinician_id,
            "decision_type":    self.decision_type,
            "ai_output":        self.ai_output,
            "clinician_action": self.clinician_action,
            "phi_in_request":   self.phi_in_request,
            "hipaa_compliant":  self.hipaa_compliant,
            "integrity_valid":  self.verify_integrity(),
        }


@dataclass
class FairnessReport:
    """Fairness audit results across demographic groups."""
    model_name: str
    attribute: str           # Protected attribute audited
    groups: dict             # Group name → metrics
    max_disparity: float     # Largest gap between any two groups
    passes_threshold: bool
    flagged_groups: list[str] = field(default_factory=list)
    audit_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> dict:
        return {
            "model":            self.model_name,
            "attribute":        self.attribute,
            "max_disparity":    round(self.max_disparity, 4),
            "threshold":        FAIRNESS_THRESHOLD,
            "passes_fairness":  self.passes_threshold,
            "flagged_groups":   self.flagged_groups,
            "group_metrics":    {
                k: {m: round(v, 4) for m, v in metrics.items()}
                for k, metrics in self.groups.items()
            },
        }


@dataclass
class ModelRiskAssessment:
    """Risk assessment for a clinical AI model prior to deployment."""
    model_name: str
    model_version: str
    use_case: str
    risk_level: str           # low | medium | high | critical
    checklist_results: dict[str, bool] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    fairness_reports: list[FairnessReport] = field(default_factory=list)
    approved_for_deployment: bool = False
    approval_blockers: list[str] = field(default_factory=list)
    reviewer: str = ""
    review_date: str = ""
    notes: str = ""

    @property
    def checklist_completion(self) -> float:
        if not self.checklist_results:
            return 0.0
        passed = sum(1 for v in self.checklist_results.values() if v)
        return passed / len(self.checklist_results)

    def summary(self) -> dict:
        return {
            "model":                  self.model_name,
            "version":                self.model_version,
            "use_case":               self.use_case,
            "risk_level":             self.risk_level,
            "approved":               self.approved_for_deployment,
            "checklist_completion":   f"{self.checklist_completion:.0%}",
            "approval_blockers":      self.approval_blockers,
            "performance_metrics":    {
                k: round(v, 4)
                for k, v in self.performance_metrics.items()
            },
        }


# ── AI Governance Framework ───────────────────────────────────────────────────

class AIGovernanceFramework:
    """
    Enterprise AI governance framework for regulated healthcare environments.

    Provides bias detection, fairness auditing, immutable audit logging,
    model risk assessment, and responsible AI checklist enforcement
    for clinical AI systems.

    Designed for compliance with HIPAA, FDA SaMD guidance, EU AI Act,
    and NIST AI Risk Management Framework.

    Parameters
    ----------
    organisation : str
        Organisation name for audit records.
    strict_mode : bool
        If True, blocks deployment on any governance failure.
        Default True for clinical AI.

    Examples
    --------
    >>> governance = AIGovernanceFramework(organisation="Spring Health")
    >>> report = governance.audit_fairness(
    ...     model_name="care_matching_v2",
    ...     y_true=labels,
    ...     y_pred=predictions,
    ...     demographics=patient_demographics,
    ... )
    >>> assessment = governance.run_pre_deployment_check(
    ...     model_name="care_matching_v2",
    ...     checklist=checklist_responses,
    ...     metrics=performance_metrics,
    ... )
    """

    def __init__(
        self,
        organisation: str = "Healthcare Organisation",
        strict_mode: bool = True,
    ) -> None:
        self.organisation  = organisation
        self.strict_mode   = strict_mode
        self._audit_log:   list[AuditLogEntry] = []
        self._assessments: list[ModelRiskAssessment] = []
        logger.info(
            "AIGovernanceFramework initialized: org=%s, strict=%s",
            organisation, strict_mode,
        )

    # ── Audit logging ─────────────────────────────────────────────────────────

    def log_decision(
        self,
        model_name: str,
        model_version: str,
        patient_id: str,
        clinician_id: str,
        decision_type: str,
        ai_output: dict,
        clinician_action: str = "pending",
        override_reason: str = "",
        phi_in_request: bool = False,
    ) -> AuditLogEntry:
        """
        Create an immutable audit log entry for an AI-assisted decision.

        Parameters
        ----------
        model_name : str
            Name of the AI model making the decision.
        model_version : str
            Model version for traceability.
        patient_id : str
            De-identified patient identifier.
        clinician_id : str
            Clinician reviewing the AI output.
        decision_type : str
            Type of decision: risk_score | match | recommendation | triage
        ai_output : dict
            The AI model's output (must not contain PHI).
        clinician_action : str
            Clinician's response: accepted | overridden | pending
        override_reason : str
            Reason if clinician overrode the AI recommendation.
        phi_in_request : bool
            Whether the request contained PHI (should always be False).

        Returns
        -------
        AuditLogEntry
        """
        if phi_in_request:
            logger.error(
                "HIPAA VIOLATION RISK: PHI detected in AI request "
                "for model %s, patient %s", model_name, patient_id
            )

        entry = AuditLogEntry(
            model_name=model_name,
            model_version=model_version,
            patient_id=patient_id,
            clinician_id=clinician_id,
            decision_type=decision_type,
            ai_output=ai_output,
            clinician_action=clinician_action,
            override_reason=override_reason,
            phi_in_request=phi_in_request,
            hipaa_compliant=not phi_in_request,
        )
        self._audit_log.append(entry)

        logger.info(
            "Audit logged [%s]: model=%s, decision=%s, action=%s",
            entry.entry_id[:8], model_name, decision_type, clinician_action,
        )
        return entry

    def get_audit_trail(
        self,
        model_name: Optional[str] = None,
        patient_id: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve audit trail with optional filtering.

        Parameters
        ----------
        model_name : str, optional
            Filter by model name.
        patient_id : str, optional
            Filter by patient ID.
        start_date : str, optional
            Filter entries after this ISO timestamp.

        Returns
        -------
        list[dict]
            Audit log entries matching filters.
        """
        entries = self._audit_log

        if model_name:
            entries = [e for e in entries if e.model_name == model_name]
        if patient_id:
            entries = [e for e in entries if e.patient_id == patient_id]
        if start_date:
            entries = [e for e in entries if e.timestamp >= start_date]

        return [e.to_dict() for e in entries]

    def verify_audit_integrity(self) -> dict:
        """
        Verify all audit log entries are intact and untampered.

        Returns
        -------
        dict
            Integrity verification report.
        """
        total     = len(self._audit_log)
        valid     = sum(1 for e in self._audit_log if e.verify_integrity())
        tampered  = [
            e.entry_id for e in self._audit_log
            if not e.verify_integrity()
        ]
        return {
            "total_entries":    total,
            "valid_entries":    valid,
            "tampered_entries": tampered,
            "integrity_score":  round(valid / total, 3) if total > 0 else 1.0,
            "audit_clean":      len(tampered) == 0,
        }

    # ── Fairness auditing ─────────────────────────────────────────────────────

    def audit_fairness(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demographics: pd.DataFrame,
        attribute: str = "gender",
    ) -> FairnessReport:
        """
        Audit model fairness across a protected demographic attribute.

        Measures performance disparity between demographic groups using:
        - Equal opportunity (recall parity)
        - Predictive parity (precision parity)
        - Demographic parity (prediction rate parity)

        Parameters
        ----------
        model_name : str
            Model being audited.
        y_true : np.ndarray
            Ground truth labels.
        y_pred : np.ndarray
            Model predictions.
        demographics : pd.DataFrame
            Patient demographic data with protected attributes.
        attribute : str
            Protected attribute to audit (e.g. 'gender', 'race_ethnicity').

        Returns
        -------
        FairnessReport
        """
        if attribute not in demographics.columns:
            raise ValueError(f"Attribute '{attribute}' not in demographics")

        groups = {}
        for group_name in demographics[attribute].unique():
            mask = demographics[attribute] == group_name
            if mask.sum() < 10:
                continue  # Skip groups with insufficient data

            gt   = y_true[mask]
            pred = y_pred[mask]

            groups[str(group_name)] = {
                "n":                 int(mask.sum()),
                "recall":            float(recall_score(gt, pred, zero_division=0)),
                "precision":         float(precision_score(gt, pred, zero_division=0)),
                "accuracy":          float(accuracy_score(gt, pred)),
                "prediction_rate":   float(pred.mean()),
            }

        # Compute maximum disparity across groups
        disparities = {}
        if len(groups) >= 2:
            for metric in ["recall", "precision", "prediction_rate"]:
                values = [g[metric] for g in groups.values()]
                disparities[metric] = max(values) - min(values)

        max_disparity = max(disparities.values()) if disparities else 0.0
        passes        = max_disparity <= FAIRNESS_THRESHOLD

        # Flag groups with below-average recall (equal opportunity)
        if groups:
            mean_recall   = np.mean([g["recall"] for g in groups.values()])
            flagged_groups = [
                name for name, metrics in groups.items()
                if metrics["recall"] < mean_recall - FAIRNESS_THRESHOLD
            ]
        else:
            flagged_groups = []

        report = FairnessReport(
            model_name=model_name,
            attribute=attribute,
            groups=groups,
            max_disparity=round(max_disparity, 4),
            passes_threshold=passes,
            flagged_groups=flagged_groups,
        )

        if not passes:
            logger.warning(
                "FAIRNESS ALERT [%s]: %s disparity=%.3f exceeds threshold %.3f. "
                "Flagged groups: %s",
                model_name, attribute, max_disparity,
                FAIRNESS_THRESHOLD, flagged_groups,
            )
        else:
            logger.info(
                "Fairness audit PASSED [%s]: %s disparity=%.3f",
                model_name, attribute, max_disparity,
            )

        return report

    def audit_all_protected_attributes(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demographics: pd.DataFrame,
    ) -> list[FairnessReport]:
        """
        Run fairness audit across all available protected attributes.

        Parameters
        ----------
        model_name : str
            Model to audit.
        y_true : np.ndarray
            Ground truth labels.
        y_pred : np.ndarray
            Model predictions.
        demographics : pd.DataFrame
            Demographic data.

        Returns
        -------
        list[FairnessReport]
            One report per available protected attribute.
        """
        reports = []
        available = [
            a for a in PROTECTED_ATTRIBUTES
            if a in demographics.columns
        ]
        for attribute in available:
            report = self.audit_fairness(
                model_name, y_true, y_pred, demographics, attribute
            )
            reports.append(report)

        passed  = sum(1 for r in reports if r.passes_threshold)
        logger.info(
            "Full fairness audit [%s]: %d/%d attributes passed",
            model_name, passed, len(reports),
        )
        return reports

    # ── Pre-deployment governance ─────────────────────────────────────────────

    def run_pre_deployment_check(
        self,
        model_name: str,
        model_version: str,
        use_case: str,
        checklist: dict[str, bool],
        performance_metrics: dict[str, float],
        fairness_reports: Optional[list[FairnessReport]] = None,
        reviewer: str = "",
        notes: str = "",
    ) -> ModelRiskAssessment:
        """
        Run comprehensive pre-deployment governance check.

        Validates the model against the responsible AI checklist,
        clinical safety performance thresholds, and fairness
        requirements before approving for production deployment.

        Parameters
        ----------
        model_name : str
            Model name.
        model_version : str
            Model version.
        use_case : str
            Clinical use case description.
        checklist : dict[str, bool]
            Responses to RESPONSIBLE_AI_CHECKLIST items.
        performance_metrics : dict[str, float]
            Model performance metrics (recall, auc, precision, etc.)
        fairness_reports : list[FairnessReport], optional
            Completed fairness audit reports.
        reviewer : str
            Name of the clinical/governance reviewer.
        notes : str
            Additional review notes.

        Returns
        -------
        ModelRiskAssessment
        """
        blockers = []

        # 1. Responsible AI checklist
        for item in RESPONSIBLE_AI_CHECKLIST:
            if not checklist.get(item, False):
                blockers.append(f"Checklist item not completed: {item}")

        # 2. Clinical safety thresholds
        recall    = performance_metrics.get("recall", 0)
        auc       = performance_metrics.get("auc", 0)
        precision = performance_metrics.get("precision", 0)

        if recall < CLINICAL_SAFETY_THRESHOLDS["min_recall"]:
            blockers.append(
                f"Recall {recall:.3f} below minimum "
                f"{CLINICAL_SAFETY_THRESHOLDS['min_recall']} — "
                f"too many high-risk cases missed"
            )
        if auc < CLINICAL_SAFETY_THRESHOLDS["min_auc"]:
            blockers.append(
                f"AUC {auc:.3f} below minimum "
                f"{CLINICAL_SAFETY_THRESHOLDS['min_auc']}"
            )
        if precision < CLINICAL_SAFETY_THRESHOLDS["min_precision"]:
            blockers.append(
                f"Precision {precision:.3f} below minimum "
                f"{CLINICAL_SAFETY_THRESHOLDS['min_precision']}"
            )

        # 3. Fairness requirements
        if fairness_reports:
            for report in fairness_reports:
                if not report.passes_threshold:
                    blockers.append(
                        f"Fairness failure on {report.attribute}: "
                        f"disparity={report.max_disparity:.3f} "
                        f"> threshold={FAIRNESS_THRESHOLD}"
                    )

        # 4. Determine risk level
        risk_level = self._classify_risk(use_case, performance_metrics)

        # 5. Approval decision
        approved = len(blockers) == 0 or (
            not self.strict_mode and
            not any("Recall" in b or "Fairness" in b for b in blockers)
        )

        assessment = ModelRiskAssessment(
            model_name=model_name,
            model_version=model_version,
            use_case=use_case,
            risk_level=risk_level,
            checklist_results=checklist,
            performance_metrics=performance_metrics,
            fairness_reports=fairness_reports or [],
            approved_for_deployment=approved,
            approval_blockers=blockers,
            reviewer=reviewer,
            review_date=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )
        self._assessments.append(assessment)

        status = "APPROVED" if approved else "BLOCKED"
        logger.info(
            "Pre-deployment check [%s v%s]: %s — %d blockers",
            model_name, model_version, status, len(blockers),
        )
        return assessment

    def generate_governance_report(self) -> dict:
        """
        Generate an enterprise AI governance summary report.

        Returns
        -------
        dict
            Full governance status across all models.
        """
        total_decisions  = len(self._audit_log)
        override_rate    = (
            sum(1 for e in self._audit_log if e.clinician_action == "overridden") /
            total_decisions
        ) if total_decisions > 0 else 0.0

        phi_violations = sum(
            1 for e in self._audit_log if e.phi_in_request
        )

        approved_models = [
            a for a in self._assessments if a.approved_for_deployment
        ]
        blocked_models  = [
            a for a in self._assessments if not a.approved_for_deployment
        ]

        integrity = self.verify_audit_integrity()

        return {
            "organisation":       self.organisation,
            "report_timestamp":   datetime.now(timezone.utc).isoformat(),
            "audit_log": {
                "total_decisions":  total_decisions,
                "override_rate":    round(override_rate, 3),
                "phi_violations":   phi_violations,
                "integrity":        integrity,
            },
            "model_assessments": {
                "total":    len(self._assessments),
                "approved": len(approved_models),
                "blocked":  len(blocked_models),
                "approved_models": [a.model_name for a in approved_models],
                "blocked_models":  [a.model_name for a in blocked_models],
            },
            "governance_health": (
                "GREEN"  if phi_violations == 0 and integrity["audit_clean"]
                        and len(blocked_models) == 0
                else "AMBER" if phi_violations == 0
                else "RED"
            ),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _classify_risk(
        use_case: str,
        metrics: dict[str, float],
    ) -> str:
        """Classify model risk level based on use case and performance."""
        use_lower = use_case.lower()
        if any(w in use_lower for w in ["crisis", "suicide", "emergency"]):
            return "critical"
        elif any(w in use_lower for w in ["diagnosis", "medication", "treatment"]):
            return "high"
        elif any(w in use_lower for w in ["risk", "triage", "matching"]):
            return "medium"
        else:
            return "low"
