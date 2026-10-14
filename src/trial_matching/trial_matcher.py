"""
trial_matching/trial_matcher.py
────────────────────────────────
Real-time clinical trial matching platform.

Mirrors Klinixa — the patient-centric trial matching platform
built at Takeda to enhance patient experience and reduce burden
across global clinical operations.

Matches patients to eligible clinical trials by evaluating:
    - Inclusion/exclusion criteria against FHIR patient profiles
    - Biomarker and genomic eligibility (companion diagnostics)
    - Geographic proximity and site availability
    - Prior treatment history and washout periods
    - Patient preference and burden scoring

Clinical standards:
    - HL7 FHIR R4 patient profiles
    - CDISC CDASH eligibility criteria
    - ClinicalTrials.gov protocol format
    - ICH E6(R2) GCP eligibility requirements

Author : Girish Rajeev
         Senior Director, AI & Clinical Data Platforms
         Former Takeda Alta Petens ($125M AI Transformation)
         CDISC CORE Board Member | Klinixa Platform Architect
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Trial eligibility criteria types ─────────────────────────────────────────

CRITERION_TYPES = {
    "age":              "Patient age range requirement",
    "diagnosis":        "Required ICD-10 diagnosis code(s)",
    "biomarker":        "Molecular biomarker or genomic alteration",
    "prior_treatment":  "Prior therapy requirement or exclusion",
    "performance":      "ECOG/Karnofsky performance status",
    "lab_value":        "Laboratory value within specified range",
    "organ_function":   "Organ function requirement (renal, hepatic, cardiac)",
    "washout":          "Washout period from prior therapy",
    "comorbidity":      "Comorbidity inclusion or exclusion",
    "geographic":       "Trial site geographic requirement",
}

# ECOG performance status labels
ECOG_LABELS = {
    0: "Fully active",
    1: "Restricted but ambulatory",
    2: "Ambulatory, self-care only",
    3: "Limited self-care",
    4: "Completely disabled",
}

# Match status
MATCH_STATUS = {
    "ELIGIBLE":          "Patient meets all inclusion/exclusion criteria",
    "POTENTIALLY":       "Patient potentially eligible — manual review needed",
    "INELIGIBLE":        "Patient does not meet one or more criteria",
    "INSUFFICIENT_DATA": "Insufficient patient data to determine eligibility",
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class EligibilityCriterion:
    """A single eligibility criterion from a trial protocol."""
    criterion_id: str
    criterion_type: str           # From CRITERION_TYPES
    description: str
    is_inclusion: bool            # True = inclusion, False = exclusion
    is_mandatory: bool = True     # Mandatory vs optional/preferred
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    required_values: list[str] = field(default_factory=list)
    excluded_values: list[str] = field(default_factory=list)
    washout_days: Optional[int] = None


@dataclass
class ClinicalTrial:
    """A clinical trial with eligibility criteria."""
    trial_id: str                 # NCT number or internal ID
    title: str
    phase: str                    # Phase 1, 2, 3, 4
    indication: str               # Primary disease/condition
    sponsor: str
    status: str                   # Recruiting, Active, Completed
    inclusion_criteria: list[EligibilityCriterion] = field(default_factory=list)
    exclusion_criteria: list[EligibilityCriterion] = field(default_factory=list)
    sites: list[dict] = field(default_factory=list)
    biomarker_requirements: list[str] = field(default_factory=list)
    max_enrollment: int = 0
    current_enrollment: int = 0
    primary_endpoint: str = ""
    therapeutic_area: str = ""

    @property
    def is_recruiting(self) -> bool:
        return self.status.lower() == "recruiting"

    @property
    def enrollment_available(self) -> bool:
        if self.max_enrollment == 0:
            return True
        return self.current_enrollment < self.max_enrollment


@dataclass
class PatientProfile:
    """
    FHIR R4-aligned patient profile for trial matching.
    All fields are de-identified — no PHI stored.
    """
    patient_id: str               # De-identified

    # Demographics
    age: Optional[int] = None
    sex: Optional[str] = None     # male | female | other
    race_ethnicity: Optional[str] = None

    # Clinical
    diagnoses: list[str] = field(default_factory=list)        # ICD-10 codes
    biomarkers: list[str] = field(default_factory=list)        # Positive biomarkers
    prior_treatments: list[str] = field(default_factory=list)  # Prior therapies
    current_medications: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)     # ICD-10 codes

    # Performance and function
    ecog_status: Optional[int] = None          # 0-4
    karnofsky_score: Optional[int] = None      # 0-100

    # Lab values
    lab_values: dict[str, float] = field(default_factory=dict)

    # Geographic
    zip_code: Optional[str] = None
    country: str = "US"
    max_travel_miles: Optional[int] = None

    # Preferences
    patient_preferences: dict = field(default_factory=dict)
    willing_placebo: bool = True
    willing_iv_therapy: bool = True

    # Timing
    last_treatment_date: Optional[str] = None   # ISO date


@dataclass
class CriterionEvaluation:
    """Result of evaluating a single eligibility criterion."""
    criterion_id: str
    criterion_type: str
    description: str
    is_inclusion: bool
    met: bool
    confidence: float             # 0-1 confidence in evaluation
    reason: str
    missing_data: bool = False


@dataclass
class TrialMatchResult:
    """Result of matching a patient to a single trial."""
    trial_id: str
    trial_title: str
    patient_id: str
    match_status: str             # From MATCH_STATUS
    overall_score: float          # 0-1 composite match score
    eligibility_score: float      # Fraction of criteria met
    preference_score: float       # Patient preference alignment
    burden_score: float           # Lower = less patient burden
    n_criteria_met: int
    n_criteria_total: int
    n_criteria_missing_data: int
    criterion_evaluations: list[CriterionEvaluation] = field(default_factory=list)
    blocking_criteria: list[str] = field(default_factory=list)
    missing_data_criteria: list[str] = field(default_factory=list)
    site_distance_miles: Optional[float] = None
    rank: int = 0

    def to_dict(self) -> dict:
        return {
            "rank":                   self.rank,
            "trial_id":               self.trial_id,
            "trial_title":            self.trial_title,
            "match_status":           self.match_status,
            "overall_score":          round(self.overall_score, 3),
            "eligibility_score":      round(self.eligibility_score, 3),
            "n_criteria_met":         self.n_criteria_met,
            "n_criteria_total":       self.n_criteria_total,
            "n_missing_data":         self.n_criteria_missing_data,
            "blocking_criteria":      self.blocking_criteria,
            "missing_data_criteria":  self.missing_data_criteria,
        }


@dataclass
class MatchingReport:
    """Complete trial matching report for a patient."""
    patient_id: str
    n_trials_evaluated: int
    n_eligible: int
    n_potentially_eligible: int
    n_ineligible: int
    matches: list[TrialMatchResult]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    disclaimer: str = (
        "AI-assisted trial matching for decision support only. "
        "Final eligibility determination requires clinical review "
        "and investigator assessment per ICH E6(R2) GCP guidelines."
    )

    def eligible_trials(self) -> list[TrialMatchResult]:
        return [m for m in self.matches if m.match_status == "ELIGIBLE"]

    def top_matches(self, n: int = 5) -> list[TrialMatchResult]:
        return self.matches[:n]

    def summary(self) -> dict:
        return {
            "patient_id":            self.patient_id,
            "timestamp":             self.timestamp,
            "trials_evaluated":      self.n_trials_evaluated,
            "eligible":              self.n_eligible,
            "potentially_eligible":  self.n_potentially_eligible,
            "ineligible":            self.n_ineligible,
            "top_5_matches":         [m.to_dict() for m in self.top_matches(5)],
            "disclaimer":            self.disclaimer,
        }


# ── Trial Matcher ─────────────────────────────────────────────────────────────

class ClinicalTrialMatcher:
    """
    Real-time clinical trial matching platform.

    Mirrors Klinixa — the patient-centric trial matching platform
    built at Takeda as part of the Alta Petens AI transformation.

    Evaluates patient eligibility against trial inclusion/exclusion
    criteria using FHIR R4 patient profiles, with scoring across
    eligibility, patient preference, and burden dimensions.

    Parameters
    ----------
    fuzzy_matching : bool
        Allow partial diagnosis code matching (e.g. C34 matches C34.1).
        Default True.
    min_confidence : float
        Minimum confidence threshold for criterion evaluation.
        Default 0.6.

    Examples
    --------
    >>> matcher = ClinicalTrialMatcher()
    >>> report = matcher.match_patient(
    ...     patient=patient_profile,
    ...     trials=trial_database,
    ... )
    >>> for match in report.top_matches(3):
    ...     print(match.trial_id, match.match_status, match.overall_score)
    """

    WEIGHTS = {
        "eligibility": 0.60,
        "preference":  0.25,
        "burden":      0.15,
    }

    def __init__(
        self,
        fuzzy_matching: bool = True,
        min_confidence: float = 0.6,
    ) -> None:
        self.fuzzy_matching  = fuzzy_matching
        self.min_confidence  = min_confidence
        logger.info(
            "ClinicalTrialMatcher initialized: fuzzy=%s, min_conf=%.2f",
            fuzzy_matching, min_confidence,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def match_patient(
        self,
        patient: PatientProfile,
        trials: list[ClinicalTrial],
        max_results: int = 10,
    ) -> MatchingReport:
        """
        Match a patient against a trial portfolio.

        Evaluates each trial's inclusion and exclusion criteria
        against the patient's FHIR profile, scores matches across
        eligibility, preference, and burden dimensions, and returns
        a ranked list of trial matches.

        Parameters
        ----------
        patient : PatientProfile
            De-identified patient profile.
        trials : list[ClinicalTrial]
            Trial portfolio to match against.
        max_results : int
            Maximum matches to return. Default 10.

        Returns
        -------
        MatchingReport
        """
        results = []

        # Filter to recruiting trials with available slots
        active_trials = [
            t for t in trials
            if t.is_recruiting and t.enrollment_available
        ]

        logger.info(
            "Matching patient %s against %d active trials",
            patient.patient_id, len(active_trials),
        )

        for trial in active_trials:
            result = self._evaluate_trial(patient, trial)
            results.append(result)

        # Sort by overall score
        results.sort(key=lambda r: r.overall_score, reverse=True)

        # Rank
        for i, result in enumerate(results[:max_results], 1):
            result.rank = i

        n_eligible    = sum(1 for r in results if r.match_status == "ELIGIBLE")
        n_potentially = sum(1 for r in results if r.match_status == "POTENTIALLY")
        n_ineligible  = sum(1 for r in results if r.match_status == "INELIGIBLE")

        report = MatchingReport(
            patient_id=patient.patient_id,
            n_trials_evaluated=len(active_trials),
            n_eligible=n_eligible,
            n_potentially_eligible=n_potentially,
            n_ineligible=n_ineligible,
            matches=results[:max_results],
        )

        logger.info(
            "Match complete [%s]: %d eligible, %d potential, %d ineligible",
            patient.patient_id, n_eligible, n_potentially, n_ineligible,
        )
        return report

    def explain_match(
        self,
        patient: PatientProfile,
        trial: ClinicalTrial,
    ) -> dict:
        """
        Generate a detailed explanation of a patient-trial match.

        Parameters
        ----------
        patient : PatientProfile
            Patient profile.
        trial : ClinicalTrial
            Trial to explain.

        Returns
        -------
        dict
            Detailed breakdown of each criterion evaluation.
        """
        result = self._evaluate_trial(patient, trial)

        explanation = {
            "trial_id":      trial.trial_id,
            "trial_title":   trial.title,
            "patient_id":    patient.patient_id,
            "match_status":  result.match_status,
            "overall_score": round(result.overall_score, 3),
            "criteria_detail": [],
        }

        for eval in result.criterion_evaluations:
            explanation["criteria_detail"].append({
                "criterion_id":   eval.criterion_id,
                "type":           eval.criterion_type,
                "description":    eval.description,
                "is_inclusion":   eval.is_inclusion,
                "met":            eval.met,
                "confidence":     round(eval.confidence, 2),
                "reason":         eval.reason,
                "missing_data":   eval.missing_data,
            })

        return explanation

    def find_near_misses(
        self,
        patient: PatientProfile,
        trials: list[ClinicalTrial],
        max_blocking: int = 2,
    ) -> list[TrialMatchResult]:
        """
        Find trials where patient is close to eligible.

        Returns trials where the patient fails only 1-2 criteria —
        useful for identifying protocol amendments or future eligibility.

        Parameters
        ----------
        max_blocking : int
            Maximum blocking criteria to qualify as near-miss. Default 2.

        Returns
        -------
        list[TrialMatchResult]
            Near-miss trials sorted by number of blocking criteria.
        """
        report = self.match_patient(patient, trials, max_results=len(trials))

        near_misses = [
            r for r in report.matches
            if r.match_status == "INELIGIBLE" and
            len(r.blocking_criteria) <= max_blocking
        ]

        near_misses.sort(key=lambda r: len(r.blocking_criteria))

        logger.info(
            "Near-miss analysis [%s]: %d trials within %d criteria",
            patient.patient_id, len(near_misses), max_blocking,
        )
        return near_misses

    # ── Private helpers ───────────────────────────────────────────────────────

    def _evaluate_trial(
        self,
        patient: PatientProfile,
        trial: ClinicalTrial,
    ) -> TrialMatchResult:
        """Evaluate a patient against a single trial."""
        evaluations      = []
        blocking_criteria = []
        missing_data_criteria = []

        # Evaluate all criteria
        all_criteria = (
            [(c, True)  for c in trial.inclusion_criteria] +
            [(c, False) for c in trial.exclusion_criteria]
        )

        for criterion, is_inclusion in all_criteria:
            evaluation = self._evaluate_criterion(patient, criterion)
            evaluations.append(evaluation)

            if evaluation.missing_data:
                missing_data_criteria.append(criterion.criterion_id)
            elif is_inclusion and not evaluation.met and criterion.is_mandatory:
                blocking_criteria.append(criterion.criterion_id)
            elif not is_inclusion and evaluation.met and criterion.is_mandatory:
                # Exclusion criterion met = blocking
                blocking_criteria.append(criterion.criterion_id)

        # Compute scores
        eligibility_score = self._compute_eligibility_score(
            evaluations, trial
        )
        preference_score  = self._compute_preference_score(patient, trial)
        burden_score      = self._compute_burden_score(patient, trial)

        overall_score = (
            self.WEIGHTS["eligibility"] * eligibility_score +
            self.WEIGHTS["preference"]  * preference_score +
            self.WEIGHTS["burden"]      * (1 - burden_score)
        )

        # Determine match status
        n_mandatory    = sum(
            1 for c in trial.inclusion_criteria if c.is_mandatory
        ) + sum(
            1 for c in trial.exclusion_criteria if c.is_mandatory
        )
        n_missing = len(missing_data_criteria)

        if len(blocking_criteria) == 0 and n_missing == 0:
            match_status = "ELIGIBLE"
        elif len(blocking_criteria) == 0 and n_missing > 0:
            match_status = "POTENTIALLY"
        elif len(blocking_criteria) > 0:
            match_status = "INELIGIBLE"
            overall_score *= 0.3  # Penalise ineligible trials
        else:
            match_status = "INSUFFICIENT_DATA"

        n_met = sum(
            1 for e in evaluations
            if e.met and not e.missing_data
        )

        return TrialMatchResult(
            trial_id=trial.trial_id,
            trial_title=trial.title,
            patient_id=patient.patient_id,
            match_status=match_status,
            overall_score=round(overall_score, 4),
            eligibility_score=round(eligibility_score, 4),
            preference_score=round(preference_score, 4),
            burden_score=round(burden_score, 4),
            n_criteria_met=n_met,
            n_criteria_total=len(evaluations),
            n_criteria_missing_data=n_missing,
            criterion_evaluations=evaluations,
            blocking_criteria=blocking_criteria,
            missing_data_criteria=missing_data_criteria,
        )

    def _evaluate_criterion(
        self,
        patient: PatientProfile,
        criterion: EligibilityCriterion,
    ) -> CriterionEvaluation:
        """Evaluate a single criterion against the patient profile."""

        ctype = criterion.criterion_type

        if ctype == "age":
            return self._eval_age(patient, criterion)
        elif ctype == "diagnosis":
            return self._eval_diagnosis(patient, criterion)
        elif ctype == "biomarker":
            return self._eval_biomarker(patient, criterion)
        elif ctype == "prior_treatment":
            return self._eval_prior_treatment(patient, criterion)
        elif ctype == "performance":
            return self._eval_performance(patient, criterion)
        elif ctype == "lab_value":
            return self._eval_lab_value(patient, criterion)
        elif ctype == "comorbidity":
            return self._eval_comorbidity(patient, criterion)
        else:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_type=ctype,
                description=criterion.description,
                is_inclusion=criterion.is_inclusion,
                met=True,
                confidence=0.5,
                reason="Criterion type not automatically evaluated — manual review required",
                missing_data=True,
            )

    def _eval_age(
        self,
        patient: PatientProfile,
        criterion: EligibilityCriterion,
    ) -> CriterionEvaluation:
        """Evaluate age criterion."""
        if patient.age is None:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_type="age",
                description=criterion.description,
                is_inclusion=criterion.is_inclusion,
                met=False, confidence=0.0,
                reason="Patient age not available",
                missing_data=True,
            )

        met = True
        reason_parts = []

        if criterion.value_min is not None:
            if patient.age < criterion.value_min:
                met = False
                reason_parts.append(
                    f"Age {patient.age} < minimum {criterion.value_min}"
                )
            else:
                reason_parts.append(
                    f"Age {patient.age} >= minimum {criterion.value_min}"
                )

        if criterion.value_max is not None:
            if patient.age > criterion.value_max:
                met = False
                reason_parts.append(
                    f"Age {patient.age} > maximum {criterion.value_max}"
                )
            else:
                reason_parts.append(
                    f"Age {patient.age} <= maximum {criterion.value_max}"
                )

        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_type="age",
            description=criterion.description,
            is_inclusion=criterion.is_inclusion,
            met=met, confidence=1.0,
            reason=" | ".join(reason_parts) or "Age criterion met",
        )

    def _eval_diagnosis(
        self,
        patient: PatientProfile,
        criterion: EligibilityCriterion,
    ) -> CriterionEvaluation:
        """Evaluate diagnosis criterion with optional fuzzy ICD-10 matching."""
        if not patient.diagnoses:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_type="diagnosis",
                description=criterion.description,
                is_inclusion=criterion.is_inclusion,
                met=False, confidence=0.0,
                reason="No patient diagnoses available",
                missing_data=True,
            )

        required = criterion.required_values
        excluded = criterion.excluded_values

        def matches_code(patient_codes, target_code):
            for code in patient_codes:
                if code == target_code:
                    return True
                if self.fuzzy_matching and code.startswith(target_code[:3]):
                    return True
            return False

        # Check required diagnoses
        if required:
            matched = [r for r in required if matches_code(patient.diagnoses, r)]
            if not matched:
                return CriterionEvaluation(
                    criterion_id=criterion.criterion_id,
                    criterion_type="diagnosis",
                    description=criterion.description,
                    is_inclusion=criterion.is_inclusion,
                    met=False, confidence=0.9,
                    reason=f"Required diagnosis {required} not in patient record",
                )

        # Check excluded diagnoses
        if excluded:
            found_excluded = [
                e for e in excluded if matches_code(patient.diagnoses, e)
            ]
            if found_excluded:
                return CriterionEvaluation(
                    criterion_id=criterion.criterion_id,
                    criterion_type="diagnosis",
                    description=criterion.description,
                    is_inclusion=criterion.is_inclusion,
                    met=False, confidence=0.9,
                    reason=f"Excluded diagnosis found: {found_excluded}",
                )

        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_type="diagnosis",
            description=criterion.description,
            is_inclusion=criterion.is_inclusion,
            met=True, confidence=0.9,
            reason="Diagnosis criteria met",
        )

    def _eval_biomarker(
        self,
        patient: PatientProfile,
        criterion: EligibilityCriterion,
    ) -> CriterionEvaluation:
        """Evaluate biomarker criterion."""
        if not patient.biomarkers:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_type="biomarker",
                description=criterion.description,
                is_inclusion=criterion.is_inclusion,
                met=False, confidence=0.0,
                reason="Biomarker data not available",
                missing_data=True,
            )

        required = criterion.required_values
        if required:
            matched = [
                r for r in required
                if any(r.lower() in b.lower() for b in patient.biomarkers)
            ]
            if not matched:
                return CriterionEvaluation(
                    criterion_id=criterion.criterion_id,
                    criterion_type="biomarker",
                    description=criterion.description,
                    is_inclusion=criterion.is_inclusion,
                    met=False, confidence=0.85,
                    reason=f"Required biomarker {required} not detected",
                )

        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_type="biomarker",
            description=criterion.description,
            is_inclusion=criterion.is_inclusion,
            met=True, confidence=0.85,
            reason="Biomarker criterion met",
        )

    def _eval_prior_treatment(
        self,
        patient: PatientProfile,
        criterion: EligibilityCriterion,
    ) -> CriterionEvaluation:
        """Evaluate prior treatment criterion."""
        required = criterion.required_values
        excluded = criterion.excluded_values

        if required and not patient.prior_treatments:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_type="prior_treatment",
                description=criterion.description,
                is_inclusion=criterion.is_inclusion,
                met=False, confidence=0.0,
                reason="Prior treatment history not available",
                missing_data=True,
            )

        if required:
            matched = [
                r for r in required
                if any(r.lower() in t.lower() for t in patient.prior_treatments)
            ]
            if not matched:
                return CriterionEvaluation(
                    criterion_id=criterion.criterion_id,
                    criterion_type="prior_treatment",
                    description=criterion.description,
                    is_inclusion=criterion.is_inclusion,
                    met=False, confidence=0.8,
                    reason=f"Required prior treatment {required} not in history",
                )

        if excluded:
            found = [
                e for e in excluded
                if any(e.lower() in t.lower() for t in patient.prior_treatments)
            ]
            if found:
                return CriterionEvaluation(
                    criterion_id=criterion.criterion_id,
                    criterion_type="prior_treatment",
                    description=criterion.description,
                    is_inclusion=criterion.is_inclusion,
                    met=False, confidence=0.85,
                    reason=f"Excluded prior treatment found: {found}",
                )

        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_type="prior_treatment",
            description=criterion.description,
            is_inclusion=criterion.is_inclusion,
            met=True, confidence=0.8,
            reason="Prior treatment criterion met",
        )

    def _eval_performance(
        self,
        patient: PatientProfile,
        criterion: EligibilityCriterion,
    ) -> CriterionEvaluation:
        """Evaluate ECOG performance status criterion."""
        if patient.ecog_status is None:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_type="performance",
                description=criterion.description,
                is_inclusion=criterion.is_inclusion,
                met=False, confidence=0.0,
                reason="ECOG performance status not recorded",
                missing_data=True,
            )

        met = True
        reason = f"ECOG {patient.ecog_status}"

        if criterion.value_max is not None:
            if patient.ecog_status > criterion.value_max:
                met = False
                reason = (
                    f"ECOG {patient.ecog_status} exceeds "
                    f"maximum {int(criterion.value_max)} "
                    f"({ECOG_LABELS.get(patient.ecog_status, '')})"
                )
            else:
                reason = (
                    f"ECOG {patient.ecog_status} within "
                    f"requirement (<= {int(criterion.value_max)})"
                )

        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_type="performance",
            description=criterion.description,
            is_inclusion=criterion.is_inclusion,
            met=met, confidence=1.0,
            reason=reason,
        )

    def _eval_lab_value(
        self,
        patient: PatientProfile,
        criterion: EligibilityCriterion,
    ) -> CriterionEvaluation:
        """Evaluate laboratory value criterion."""
        lab_name = (
            criterion.required_values[0]
            if criterion.required_values else None
        )

        if not lab_name or lab_name not in patient.lab_values:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_type="lab_value",
                description=criterion.description,
                is_inclusion=criterion.is_inclusion,
                met=False, confidence=0.0,
                reason=f"Lab value '{lab_name}' not available",
                missing_data=True,
            )

        value = patient.lab_values[lab_name]
        met   = True
        reason_parts = [f"{lab_name} = {value:.2f}"]

        if criterion.value_min is not None and value < criterion.value_min:
            met = False
            reason_parts.append(f"< minimum {criterion.value_min}")
        if criterion.value_max is not None and value > criterion.value_max:
            met = False
            reason_parts.append(f"> maximum {criterion.value_max}")

        if met:
            reason_parts.append("within range")

        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_type="lab_value",
            description=criterion.description,
            is_inclusion=criterion.is_inclusion,
            met=met, confidence=1.0,
            reason=" | ".join(reason_parts),
        )

    def _eval_comorbidity(
        self,
        patient: PatientProfile,
        criterion: EligibilityCriterion,
    ) -> CriterionEvaluation:
        """Evaluate comorbidity criterion."""
        excluded = criterion.excluded_values

        if not excluded:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_type="comorbidity",
                description=criterion.description,
                is_inclusion=criterion.is_inclusion,
                met=True, confidence=0.7,
                reason="No specific comorbidities to exclude",
            )

        found = [
            e for e in excluded
            if any(
                e.lower() in c.lower()
                for c in patient.comorbidities
            )
        ]

        if found:
            return CriterionEvaluation(
                criterion_id=criterion.criterion_id,
                criterion_type="comorbidity",
                description=criterion.description,
                is_inclusion=criterion.is_inclusion,
                met=False, confidence=0.85,
                reason=f"Excluded comorbidity present: {found}",
            )

        return CriterionEvaluation(
            criterion_id=criterion.criterion_id,
            criterion_type="comorbidity",
            description=criterion.description,
            is_inclusion=criterion.is_inclusion,
            met=True, confidence=0.75,
            reason="No excluded comorbidities found",
        )

    def _compute_eligibility_score(
        self,
        evaluations: list[CriterionEvaluation],
        trial: ClinicalTrial,
    ) -> float:
        """Compute weighted eligibility score."""
        if not evaluations:
            return 0.0

        mandatory_evals = [
            e for e in evaluations
            if not e.missing_data
        ]

        if not mandatory_evals:
            return 0.5  # Insufficient data prior

        n_met = sum(1 for e in mandatory_evals if e.met)
        weighted_score = sum(
            e.confidence for e in mandatory_evals if e.met
        ) / (len(mandatory_evals) + 1e-10)

        return float(weighted_score)

    def _compute_preference_score(
        self,
        patient: PatientProfile,
        trial: ClinicalTrial,
    ) -> float:
        """Score alignment between patient preferences and trial characteristics."""
        score = 0.5  # Neutral baseline

        # Willingness for IV therapy
        if not patient.willing_iv_therapy:
            score -= 0.1

        # Willingness for placebo
        if not patient.willing_placebo:
            score -= 0.05

        # Phase preference (some patients prefer later-phase trials)
        if "3" in trial.phase or "4" in trial.phase:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _compute_burden_score(
        self,
        patient: PatientProfile,
        trial: ClinicalTrial,
    ) -> float:
        """
        Compute patient burden score (0 = low burden, 1 = high burden).

        Considers geographic burden, visit frequency, and
        treatment complexity.
        """
        burden = 0.3  # Baseline moderate burden

        # Geographic burden
        if patient.max_travel_miles and trial.sites:
            burden += 0.2  # Simplified — in production uses geolocation

        # Phase-based burden (Phase 1 = higher burden)
        if "1" in trial.phase:
            burden += 0.2
        elif "2" in trial.phase:
            burden += 0.1

        return max(0.0, min(1.0, burden))
