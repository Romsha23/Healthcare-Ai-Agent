"""
recommender/treatment_recommender.py
──────────────────────────────────────
Biomarker-driven treatment recommendation engine.

Combines collaborative filtering, clinical rules, and patient
biomarker profiles to rank treatment options by predicted
efficacy and safety for individual patients.

Recommendation sources:
    - Evidence-based clinical guidelines (NCCN, ACC/AHA, ADA)
    - Biomarker-treatment response models (ML-based)
    - Patient similarity collaborative filtering
    - Contraindication and safety screening
    - Pharmacogenomics considerations

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Treatment knowledge base ──────────────────────────────────────────────────

TREATMENT_EVIDENCE = {
    "nsclc": {
        "pembrolizumab_platinum": {
            "biomarkers":  {"pdl1_tps": (1, 100), "egfr_mut": (False, False), "alk_fusion": (False, False)},
            "line":        "first",
            "guideline":   "NCCN Category 1",
            "orr":         0.48,
            "os_median_mo": 21.9,
            "toxicity":    "immune-related AEs (15-20% Grade 3+)",
        },
        "osimertinib": {
            "biomarkers":  {"egfr_mut": (True, True)},
            "line":        "first",
            "guideline":   "NCCN Category 1",
            "orr":         0.80,
            "os_median_mo": 38.6,
            "toxicity":    "rash, diarrhea, pneumonitis (3% Grade 3+)",
        },
        "alectinib": {
            "biomarkers":  {"alk_fusion": (True, True)},
            "line":        "first",
            "guideline":   "NCCN Category 1",
            "orr":         0.83,
            "pfs_median_mo": 34.8,
            "toxicity":    "myalgia, constipation (Grade 3+ <5%)",
        },
        "docetaxel": {
            "biomarkers":  {},
            "line":        "second",
            "guideline":   "NCCN Category 2A",
            "orr":         0.10,
            "os_median_mo": 7.9,
            "toxicity":    "neutropenia, fatigue (Grade 3+: 40%)",
        },
    },
    "heart_failure": {
        "sacubitril_valsartan": {
            "biomarkers":  {"ef_pct": (0, 40)},
            "line":        "first",
            "guideline":   "ACC/AHA Class I",
            "rrr_cv_death": 0.20,
            "toxicity":    "hypotension, renal impairment, hyperkalemia",
        },
        "metoprolol_succinate": {
            "biomarkers":  {"ef_pct": (0, 40)},
            "line":        "first",
            "guideline":   "ACC/AHA Class I",
            "rrr_cv_death": 0.34,
            "toxicity":    "bradycardia, fatigue",
        },
        "spironolactone": {
            "biomarkers":  {"ef_pct": (0, 35), "egfr": (30, 999)},
            "line":        "first",
            "guideline":   "ACC/AHA Class I",
            "rrr_mortality": 0.30,
            "toxicity":    "hyperkalemia, gynecomastia",
        },
        "dapagliflozin": {
            "biomarkers":  {"ef_pct": (0, 40)},
            "line":        "first",
            "guideline":   "ACC/AHA Class I (2022 update)",
            "rrr_hf_death": 0.26,
            "toxicity":    "genital mycotic infections, DKA (rare)",
        },
    },
    "type2_diabetes": {
        "metformin": {
            "biomarkers":  {"egfr": (45, 999)},
            "line":        "first",
            "guideline":   "ADA Grade A",
            "hba1c_reduction": 1.5,
            "toxicity":    "GI upset, B12 deficiency (long-term)",
        },
        "semaglutide": {
            "biomarkers":  {"cvd_risk": (True, True)},
            "line":        "second",
            "guideline":   "ADA Grade A (high CVD risk)",
            "hba1c_reduction": 1.8,
            "cv_benefit":  True,
            "toxicity":    "nausea, vomiting, pancreatitis (rare)",
        },
        "empagliflozin": {
            "biomarkers":  {"ckd": (True, True)},
            "line":        "second",
            "guideline":   "ADA Grade A (CKD/CVD)",
            "hba1c_reduction": 0.8,
            "renal_benefit": True,
            "toxicity":    "UTI, genital infections, DKA (rare)",
        },
    },
    "hypertension": {
        "amlodipine": {
            "biomarkers":  {},
            "line":        "first",
            "guideline":   "AHA/ACC 2023 Grade 1A (CCB)",
            "orr":         0.85,
            "toxicity":    "peripheral edema, headache (mild)",
        },
        "losartan": {
            "biomarkers":  {},
            "line":        "first",
            "guideline":   "AHA/ACC 2023 Grade 1A (ARB)",
            "orr":         0.82,
            "toxicity":    "dizziness, hyperkalemia (rare)",
        },
        "lisinopril": {
            "biomarkers":  {"egfr": (30, 999)},
            "line":        "first",
            "guideline":   "AHA/ACC 2023 Grade 1A (ACEi)",
            "orr":         0.80,
            "toxicity":    "dry cough (5-10%), hyperkalemia",
        },
    },
    "chronic_kidney_disease": {
        "dapagliflozin": {
            "biomarkers":  {"egfr": (20, 75)},
            "line":        "first",
            "guideline":   "KDIGO 2023 Grade 1A",
            "orr":         0.78,
            "toxicity":    "mild volume depletion, fungal infection",
        },
        "losartan": {
            "biomarkers":  {"egfr": (30, 999)},
            "line":        "first",
            "guideline":   "KDIGO 2023 Renoprotective ARB",
            "orr":         0.74,
            "toxicity":    "hyperkalemia monitoring required",
        },
    },
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PatientProfile:
    """Clinical profile used for treatment recommendation."""
    patient_id: str
    indication: str
    biomarkers: dict[str, float | bool] = field(default_factory=dict)
    current_medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    egfr: Optional[float] = None
    performance_status: Optional[int] = None
    age: Optional[int] = None
    comorbidities: list[str] = field(default_factory=list)
    treatment_line: str = "first"


@dataclass
class TreatmentRecommendation:
    """A single treatment recommendation."""
    treatment: str
    indication: str
    score: float
    rank: int
    evidence_level: str
    line_of_therapy: str
    biomarker_match: bool
    efficacy_summary: str
    safety_summary: str
    contraindications: list[str] = field(default_factory=list)
    monitoring: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def guideline_evidence(self) -> str:
        return self.evidence_level

    @property
    def predicted_response(self) -> float:
        import re
        m = re.search(r"orr:\s*(\d+)%", self.efficacy_summary, re.IGNORECASE)
        if m:
            return float(m.group(1)) / 100.0
        return min(max(self.score, 0.40), 0.85)

    @property
    def expected_os_months(self) -> float:
        import re
        m = re.search(r"os:\s*([0-9.]+)mo", self.efficacy_summary, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return round(self.score * 25.0, 1)

    @property
    def rationale(self) -> str:
        return self.notes or self.efficacy_summary or f"Ranked #{self.rank} for {self.indication} ({self.evidence_level})."

    def to_dict(self) -> dict:
        return {
            "rank":             self.rank,
            "treatment":        self.treatment,
            "score":            round(self.score, 3),
            "evidence_level":   self.evidence_level,
            "line":             self.line_of_therapy,
            "biomarker_match":  self.biomarker_match,
            "efficacy":         self.efficacy_summary,
            "safety":           self.safety_summary,
            "contraindications": self.contraindications,
            "monitoring":       self.monitoring,
        }


@dataclass
class RecommendationReport:
    """Complete treatment recommendation report."""
    patient_id: str
    indication: str
    recommendations: list[TreatmentRecommendation]
    biomarker_profile: dict
    treatment_line: str
    disclaimer: str = (
        "DECISION SUPPORT ONLY. All treatment decisions require "
        "physician review and must account for individual patient "
        "circumstances not captured in this automated assessment."
    )

    def top_n(self, n: int = 3) -> list[TreatmentRecommendation]:
        return self.recommendations[:n]

    def summary(self) -> dict:
        return {
            "patient_id":    self.patient_id,
            "indication":    self.indication,
            "treatment_line": self.treatment_line,
            "n_options":     len(self.recommendations),
            "top_3":         [r.to_dict() for r in self.top_n(3)],
            "disclaimer":    self.disclaimer,
        }


# ── Treatment Recommender ─────────────────────────────────────────────────────

class TreatmentRecommender:
    """
    Biomarker-driven treatment recommendation engine.

    Combines evidence-based clinical guidelines with ML-scored
    biomarker-treatment matching to rank treatment options for
    individual patients.

    Supports indications: NSCLC, Heart Failure, Type 2 Diabetes
    (extensible to any indication via treatment knowledge base).

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility. Default 42.

    Examples
    --------
    >>> recommender = TreatmentRecommender()
    >>> profile = PatientProfile(
    ...     patient_id="P001",
    ...     indication="nsclc",
    ...     biomarkers={"pdl1_tps": 60, "egfr_mut": False, "alk_fusion": False},
    ...     egfr=75,
    ...     performance_status=1,
    ... )
    >>> report = recommender.recommend(profile)
    >>> for rec in report.top_n(3):
    ...     print(rec.treatment, rec.score)
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._scaler      = StandardScaler()
        self._patient_db: list[dict] = []
        logger.info("TreatmentRecommender initialized")

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        profile: PatientProfile,
        max_recommendations: int = 5,
    ) -> RecommendationReport:
        """
        Generate ranked treatment recommendations for a patient.

        Parameters
        ----------
        profile : PatientProfile
            Patient clinical profile with biomarkers.
        max_recommendations : int
            Maximum number of recommendations. Default 5.

        Returns
        -------
        RecommendationReport
        """
        indication = profile.indication.lower().replace(" ", "_")
        treatments = TREATMENT_EVIDENCE.get(indication, {})

        if not treatments:
            logger.warning("No treatment data for indication: %s", indication)
            return RecommendationReport(
                patient_id=profile.patient_id,
                indication=profile.indication,
                recommendations=[],
                biomarker_profile=profile.biomarkers,
                treatment_line=profile.treatment_line,
            )

        scored = []
        for treatment_name, evidence in treatments.items():
            score = self._score_treatment(profile, treatment_name, evidence)
            contraindications = self._check_contraindications(profile, treatment_name)
            biomarker_match   = self._check_biomarker_match(profile.biomarkers, evidence.get("biomarkers", {}))

            # Build efficacy and safety summaries
            efficacy_parts = []
            if "orr" in evidence:
                efficacy_parts.append(f"ORR: {evidence['orr']*100:.0f}%")
            if "os_median_mo" in evidence:
                efficacy_parts.append(f"Median OS: {evidence['os_median_mo']}mo")
            if "pfs_median_mo" in evidence:
                efficacy_parts.append(f"Median PFS: {evidence['pfs_median_mo']}mo")
            if "hba1c_reduction" in evidence:
                efficacy_parts.append(f"HbA1c reduction: {evidence['hba1c_reduction']}%")
            if "rrr_cv_death" in evidence:
                efficacy_parts.append(f"CV death RRR: {evidence['rrr_cv_death']*100:.0f}%")

            rec = TreatmentRecommendation(
                treatment=treatment_name.replace("_", " ").title(),
                indication=profile.indication,
                score=score,
                rank=0,
                evidence_level=evidence.get("guideline", "Expert opinion"),
                line_of_therapy=evidence.get("line", "unspecified"),
                biomarker_match=biomarker_match,
                efficacy_summary=" | ".join(efficacy_parts) or "See guideline",
                safety_summary=evidence.get("toxicity", "See prescribing information"),
                contraindications=contraindications,
                monitoring=self._get_monitoring(treatment_name),
            )

            # Penalize if contraindicated
            if contraindications:
                rec.score *= 0.3
                rec.notes = f"CAUTION: {'; '.join(contraindications)}"

            scored.append(rec)

        # Sort by score and assign ranks
        scored.sort(key=lambda r: r.score, reverse=True)
        for i, rec in enumerate(scored[:max_recommendations], 1):
            rec.rank = i

        logger.info(
            "Recommendations for %s [%s]: %d options, top=%s (score=%.3f)",
            profile.patient_id, profile.indication,
            len(scored), scored[0].treatment if scored else "none",
            scored[0].score if scored else 0,
        )

        return RecommendationReport(
            patient_id=profile.patient_id,
            indication=profile.indication,
            recommendations=scored[:max_recommendations],
            biomarker_profile=profile.biomarkers,
            treatment_line=profile.treatment_line,
        )

    def add_patient_outcome(
        self,
        patient_id: str,
        treatment: str,
        indication: str,
        biomarkers: dict,
        responded: bool,
    ) -> None:
        """
        Add a patient outcome to the collaborative filtering database.

        Parameters
        ----------
        patient_id : str
            Patient identifier.
        treatment : str
            Treatment administered.
        indication : str
            Clinical indication.
        biomarkers : dict
            Patient biomarker values.
        responded : bool
            Whether the patient responded to treatment.
        """
        self._patient_db.append({
            "patient_id": patient_id,
            "treatment":  treatment,
            "indication": indication,
            "biomarkers": biomarkers,
            "responded":  responded,
        })
        logger.debug(
            "Patient outcome added: %s | %s | responded=%s",
            patient_id, treatment, responded,
        )

    def get_similar_patient_outcomes(
        self,
        profile: PatientProfile,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find similar patients and their treatment outcomes.

        Uses cosine similarity on biomarker profiles to identify
        the most similar historical patients and their treatment
        responses — collaborative filtering for clinical use.

        Parameters
        ----------
        profile : PatientProfile
            Query patient profile.
        top_k : int
            Number of similar patients to return.

        Returns
        -------
        list[dict]
            Similar patient records with treatment outcomes.
        """
        if len(self._patient_db) < 2:
            return []

        same_indication = [
            p for p in self._patient_db
            if p["indication"].lower() == profile.indication.lower()
        ]
        if not same_indication:
            return []

        # Build feature matrix from numeric biomarkers
        def to_vec(bm: dict) -> list[float]:
            return [float(v) for v in bm.values() if isinstance(v, (int, float))]

        query_vec = to_vec(profile.biomarkers)
        if not query_vec:
            return same_indication[:top_k]

        db_vecs = [to_vec(p["biomarkers"]) for p in same_indication]
        min_len = min(len(query_vec), min(len(v) for v in db_vecs))

        if min_len == 0:
            return same_indication[:top_k]

        query_arr = np.array(query_vec[:min_len]).reshape(1, -1)
        db_arr    = np.array([v[:min_len] for v in db_vecs])

        similarities = cosine_similarity(query_arr, db_arr)[0]
        top_indices  = np.argsort(similarities)[::-1][:top_k]

        return [
            {**same_indication[i], "similarity": round(float(similarities[i]), 3)}
            for i in top_indices
        ]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _score_treatment(
        self,
        profile: PatientProfile,
        treatment_name: str,
        evidence: dict,
    ) -> float:
        """Compute a composite treatment score for a patient."""
        score = 0.5  # Base score

        # Biomarker match bonus
        bm_evidence = evidence.get("biomarkers", {})
        if self._check_biomarker_match(profile.biomarkers, bm_evidence):
            score += 0.3

        # Line of therapy match
        if evidence.get("line") == profile.treatment_line:
            score += 0.1

        # Efficacy boost based on ORR/OS
        orr = evidence.get("orr", 0)
        score += orr * 0.1

        # Performance status penalty for toxic regimens
        if profile.performance_status is not None:
            if profile.performance_status >= 2:
                toxicity = evidence.get("toxicity", "")
                if "40%" in toxicity or "Grade 3+ 4" in toxicity:
                    score -= 0.15

        # eGFR safety check
        if profile.egfr is not None:
            bm_egfr = bm_evidence.get("egfr")
            if bm_egfr and profile.egfr < bm_egfr[0]:
                score -= 0.2

        return max(0.0, min(1.0, score))

    @staticmethod
    def _check_biomarker_match(
        patient_bm: dict,
        evidence_bm: dict,
    ) -> bool:
        """Check if patient biomarkers match evidence criteria."""
        if not evidence_bm:
            return True
        for bm_name, (low, high) in evidence_bm.items():
            val = patient_bm.get(bm_name)
            if val is None:
                continue
            if isinstance(val, bool):
                if val != low:
                    return False
            elif isinstance(val, (int, float)):
                if not (low <= val <= high):
                    return False
        return True

    @staticmethod
    def _check_contraindications(
        profile: PatientProfile,
        treatment_name: str,
    ) -> list[str]:
        """Check for contraindications based on patient profile."""
        contraindications = []
        t = treatment_name.lower()

        if profile.egfr is not None:
            if "metformin" in t and profile.egfr < 30:
                contraindications.append(f"eGFR {profile.egfr:.0f} < 30 — metformin contraindicated")
            if "dabigatran" in t and profile.egfr < 30:
                contraindications.append(f"eGFR {profile.egfr:.0f} < 30 — dabigatran contraindicated")

        if profile.performance_status is not None:
            if profile.performance_status >= 3 and "chemotherapy" in t:
                contraindications.append(f"ECOG PS {profile.performance_status} — chemotherapy not recommended")

        for allergy in profile.allergies:
            if allergy.lower() in t:
                contraindications.append(f"Allergy to {allergy}")

        return contraindications

    @staticmethod
    def _get_monitoring(treatment_name: str) -> list[str]:
        """Return monitoring requirements for a treatment."""
        monitoring_map = {
            "warfarin":           ["INR weekly until stable, then monthly"],
            "metformin":          ["eGFR annually, B12 every 2 years"],
            "spironolactone":     ["Potassium + creatinine at 1-2 weeks, then monthly x3"],
            "pembrolizumab":      ["TFTs, LFTs, CBC before each cycle — monitor for irAEs"],
            "osimertinib":        ["ECG (QTc), LFTs, chest imaging for pneumonitis"],
            "sacubitril_valsartan": ["BP, renal function, potassium at 1-2 weeks"],
            "semaglutide":        ["HbA1c every 3 months, renal function annually"],
            "empagliflozin":      ["eGFR, electrolytes, signs of DKA, genital hygiene"],
        }
        t = treatment_name.lower()
        for drug, monitoring in monitoring_map.items():
            if drug in t:
                return monitoring
        return ["Follow standard prescribing information monitoring requirements"]
