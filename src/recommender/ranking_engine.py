"""
recommender/ranking_engine.py
──────────────────────────────
Multi-criteria treatment ranking engine for clinical decision support.

Implements a weighted multi-criteria decision analysis (MCDA) framework
for ranking treatment options incorporating:
    - Clinical efficacy (survival, response rates)
    - Safety profile (toxicity grade, discontinuation rates)
    - Patient-specific biomarker fit
    - Quality of life impact
    - Cost-effectiveness (where data available)
    - Patient preference weights

Methods:
    WeightedScoringRanker  — Configurable weighted scoring
    TORSISRanker           — Technique for Order Preference by Similarity (TOPSIS)
    EnsembleRanker         — Ensemble of multiple ranking methods

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Ranking criteria weights ──────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "efficacy":         0.35,   # OS/PFS/ORR benefit
    "safety":           0.25,   # Toxicity profile
    "biomarker_fit":    0.20,   # Biomarker match score
    "qol_impact":       0.10,   # Quality of life
    "guideline_level":  0.10,   # Evidence strength
}

# Weights for patient-centered preferences
PATIENT_CENTERED_WEIGHTS = {
    "efficacy":         0.25,
    "safety":           0.35,   # Safety prioritized
    "biomarker_fit":    0.20,
    "qol_impact":       0.15,   # QoL prioritized
    "guideline_level":  0.05,
}

# Evidence level numeric scores
EVIDENCE_SCORES = {
    "category 1":      1.0,
    "class i":         1.0,
    "grade a":         1.0,
    "category 2a":     0.8,
    "class iia":       0.8,
    "grade b":         0.8,
    "category 2b":     0.6,
    "class iib":       0.6,
    "grade c":         0.6,
    "expert opinion":  0.4,
    "category 3":      0.3,
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TreatmentCriteria:
    """Multi-criteria scores for a single treatment option."""
    treatment_name: str
    efficacy_score: float = 0.0       # 0-1, higher is better
    safety_score: float = 0.0         # 0-1, higher = safer
    biomarker_fit_score: float = 0.0  # 0-1, higher = better match
    qol_score: float = 0.0            # 0-1, higher = better QoL
    guideline_score: float = 0.0      # 0-1, higher = stronger evidence
    raw_data: dict = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.efficacy_score,
            self.safety_score,
            self.biomarker_fit_score,
            self.qol_score,
            self.guideline_score,
        ])


@dataclass
class RankedTreatment:
    """A treatment with its final composite rank and score."""
    rank: int
    treatment_name: str
    composite_score: float
    ranking_method: str
    criteria_scores: dict[str, float] = field(default_factory=dict)
    confidence: str = "medium"   # high | medium | low
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "rank":            self.rank,
            "treatment":       self.treatment_name,
            "composite_score": round(self.composite_score, 3),
            "method":          self.ranking_method,
            "criteria":        {k: round(v, 3) for k, v in self.criteria_scores.items()},
            "confidence":      self.confidence,
            "rationale":       self.rationale,
        }


@dataclass
class RankingResult:
    """Complete ranking result from the engine."""
    patient_id: str
    indication: str
    method: str
    ranked_treatments: list[RankedTreatment]
    weights_used: dict[str, float]
    n_treatments_evaluated: int = 0

    def top_n(self, n: int = 3) -> list[RankedTreatment]:
        return self.ranked_treatments[:n]

    def summary(self) -> dict:
        return {
            "patient_id":   self.patient_id,
            "indication":   self.indication,
            "method":       self.method,
            "n_evaluated":  self.n_treatments_evaluated,
            "top_3":        [t.to_dict() for t in self.top_n(3)],
            "weights_used": self.weights_used,
        }


# ── Weighted Scoring Ranker ───────────────────────────────────────────────────

class WeightedScoringRanker:
    """
    Configurable weighted multi-criteria scoring ranker.

    Computes a composite score for each treatment as a weighted
    sum of normalized criteria scores. Supports custom weight
    profiles for different clinical contexts.

    Parameters
    ----------
    weights : dict, optional
        Criteria weights. Must sum to 1.0.
        Default: DEFAULT_WEIGHTS (efficacy-focused).
    patient_centered : bool
        If True, uses PATIENT_CENTERED_WEIGHTS.
        Overrides weights parameter. Default False.

    Examples
    --------
    >>> ranker = WeightedScoringRanker(patient_centered=True)
    >>> criteria = [
    ...     TreatmentCriteria("Drug A", efficacy_score=0.9, safety_score=0.7,
    ...                        biomarker_fit_score=1.0, qol_score=0.8,
    ...                        guideline_score=1.0),
    ...     TreatmentCriteria("Drug B", efficacy_score=0.6, safety_score=0.9,
    ...                        biomarker_fit_score=0.5, qol_score=0.9,
    ...                        guideline_score=0.8),
    ... ]
    >>> result = ranker.rank("P001", "nsclc", criteria)
    >>> print(result.top_n(2))
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        patient_centered: bool = False,
    ) -> None:
        if patient_centered:
            self.weights = PATIENT_CENTERED_WEIGHTS.copy()
        else:
            self.weights = (weights or DEFAULT_WEIGHTS).copy()

        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Weights sum to %.3f — normalizing to 1.0", total
            )
            self.weights = {k: v / total for k, v in self.weights.items()}

    def rank(
        self,
        patient_id: str,
        indication: str,
        treatment_criteria: list[TreatmentCriteria],
    ) -> RankingResult:
        """
        Rank treatments using weighted scoring.

        Parameters
        ----------
        patient_id : str
            Patient identifier.
        indication : str
            Clinical indication.
        treatment_criteria : list[TreatmentCriteria]
            List of treatment criteria scores.

        Returns
        -------
        RankingResult
        """
        ranked = []
        for tc in treatment_criteria:
            score = (
                self.weights.get("efficacy", 0)        * tc.efficacy_score +
                self.weights.get("safety", 0)          * tc.safety_score +
                self.weights.get("biomarker_fit", 0)   * tc.biomarker_fit_score +
                self.weights.get("qol_impact", 0)      * tc.qol_score +
                self.weights.get("guideline_level", 0) * tc.guideline_score
            )

            confidence = (
                "high"   if score >= 0.75 else
                "medium" if score >= 0.50 else
                "low"
            )

            rationale = self._build_rationale(tc, score)

            ranked.append(RankedTreatment(
                rank=0,
                treatment_name=tc.treatment_name,
                composite_score=score,
                ranking_method="weighted_scoring",
                criteria_scores={
                    "efficacy":        tc.efficacy_score,
                    "safety":          tc.safety_score,
                    "biomarker_fit":   tc.biomarker_fit_score,
                    "qol_impact":      tc.qol_score,
                    "guideline_level": tc.guideline_score,
                },
                confidence=confidence,
                rationale=rationale,
            ))

        ranked.sort(key=lambda r: r.composite_score, reverse=True)
        for i, r in enumerate(ranked, 1):
            r.rank = i

        logger.info(
            "WeightedScoring ranked %d treatments for %s [%s]: top=%s (%.3f)",
            len(ranked), patient_id, indication,
            ranked[0].treatment_name if ranked else "none",
            ranked[0].composite_score if ranked else 0,
        )

        return RankingResult(
            patient_id=patient_id,
            indication=indication,
            method="weighted_scoring",
            ranked_treatments=ranked,
            weights_used=self.weights,
            n_treatments_evaluated=len(ranked),
        )

    @staticmethod
    def _build_rationale(tc: TreatmentCriteria, score: float) -> str:
        parts = []
        if tc.biomarker_fit_score >= 0.8:
            parts.append("strong biomarker match")
        if tc.efficacy_score >= 0.8:
            parts.append("high efficacy data")
        if tc.safety_score >= 0.8:
            parts.append("favorable safety profile")
        if tc.guideline_score >= 0.9:
            parts.append("Category 1/Class I evidence")
        if tc.safety_score < 0.4:
            parts.append("notable toxicity concerns")
        if tc.biomarker_fit_score < 0.3:
            parts.append("biomarker mismatch")
        return "; ".join(parts) if parts else "Moderate overall profile"


# ── TOPSIS Ranker ─────────────────────────────────────────────────────────────

class TOPSISRanker:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

    Ranks treatments by their geometric distance from the ideal
    best solution and the ideal worst solution. A treatment is
    ranked higher when it is simultaneously closest to the
    positive-ideal solution and farthest from the negative-ideal.

    Widely used in healthcare MCDA and drug formulary decisions.

    Parameters
    ----------
    weights : dict, optional
        Criteria weights. Default: DEFAULT_WEIGHTS.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def rank(
        self,
        patient_id: str,
        indication: str,
        treatment_criteria: list[TreatmentCriteria],
    ) -> RankingResult:
        """
        Rank treatments using TOPSIS.

        Parameters
        ----------
        patient_id : str
            Patient identifier.
        indication : str
            Clinical indication.
        treatment_criteria : list[TreatmentCriteria]
            List of treatment criteria scores.

        Returns
        -------
        RankingResult
        """
        if len(treatment_criteria) < 2:
            return self._single_treatment_result(
                patient_id, indication, treatment_criteria
            )

        # Build decision matrix (rows=treatments, cols=criteria)
        matrix = np.array([tc.to_vector() for tc in treatment_criteria])
        weight_vec = np.array([
            self.weights.get("efficacy", 0.2),
            self.weights.get("safety", 0.2),
            self.weights.get("biomarker_fit", 0.2),
            self.weights.get("qol_impact", 0.2),
            self.weights.get("guideline_level", 0.2),
        ])

        # Normalize decision matrix
        col_norms = np.linalg.norm(matrix, axis=0)
        col_norms[col_norms == 0] = 1.0
        norm_matrix = matrix / col_norms

        # Weighted normalized matrix
        weighted = norm_matrix * weight_vec

        # Ideal best (max for all benefit criteria) and worst (min)
        ideal_best  = np.max(weighted, axis=0)
        ideal_worst = np.min(weighted, axis=0)

        # Distance from ideal best and worst
        dist_best  = np.linalg.norm(weighted - ideal_best,  axis=1)
        dist_worst = np.linalg.norm(weighted - ideal_worst, axis=1)

        # TOPSIS closeness coefficient
        closeness = dist_worst / (dist_best + dist_worst + 1e-10)

        ranked = []
        for i, tc in enumerate(treatment_criteria):
            ranked.append(RankedTreatment(
                rank=0,
                treatment_name=tc.treatment_name,
                composite_score=float(closeness[i]),
                ranking_method="topsis",
                criteria_scores={
                    "efficacy":        tc.efficacy_score,
                    "safety":          tc.safety_score,
                    "biomarker_fit":   tc.biomarker_fit_score,
                    "qol_impact":      tc.qol_score,
                    "guideline_level": tc.guideline_score,
                    "dist_from_ideal": round(float(dist_best[i]), 4),
                    "dist_from_worst": round(float(dist_worst[i]), 4),
                },
                confidence=(
                    "high"   if closeness[i] >= 0.7 else
                    "medium" if closeness[i] >= 0.4 else
                    "low"
                ),
            ))

        ranked.sort(key=lambda r: r.composite_score, reverse=True)
        for i, r in enumerate(ranked, 1):
            r.rank = i

        logger.info(
            "TOPSIS ranked %d treatments for %s [%s]: top=%s (%.3f)",
            len(ranked), patient_id, indication,
            ranked[0].treatment_name if ranked else "none",
            ranked[0].composite_score if ranked else 0,
        )

        return RankingResult(
            patient_id=patient_id,
            indication=indication,
            method="topsis",
            ranked_treatments=ranked,
            weights_used=self.weights,
            n_treatments_evaluated=len(ranked),
        )

    def _single_treatment_result(
        self,
        patient_id: str,
        indication: str,
        treatment_criteria: list[TreatmentCriteria],
    ) -> RankingResult:
        ranked = [
            RankedTreatment(
                rank=1,
                treatment_name=tc.treatment_name,
                composite_score=float(np.mean(tc.to_vector())),
                ranking_method="topsis",
            )
            for tc in treatment_criteria
        ]
        return RankingResult(
            patient_id=patient_id,
            indication=indication,
            method="topsis",
            ranked_treatments=ranked,
            weights_used=self.weights,
            n_treatments_evaluated=len(ranked),
        )


# ── Ensemble Ranker ───────────────────────────────────────────────────────────

class EnsembleRanker:
    """
    Ensemble ranking combining WeightedScoring and TOPSIS.

    Averages rank positions from multiple ranking methods to
    produce a more robust final ranking that is less sensitive
    to the assumptions of any single method.

    Parameters
    ----------
    weights : dict, optional
        Criteria weights shared across all sub-rankers.
    patient_centered : bool
        Use patient-centered weight profile. Default False.

    Examples
    --------
    >>> ranker = EnsembleRanker()
    >>> result = ranker.rank("P001", "nsclc", criteria_list)
    >>> for t in result.top_n(3):
    ...     print(t.rank, t.treatment_name, t.composite_score)
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        patient_centered: bool = False,
    ) -> None:
        self._weighted = WeightedScoringRanker(
            weights=weights,
            patient_centered=patient_centered,
        )
        self._topsis = TOPSISRanker(
            weights=self._weighted.weights,
        )

    def rank(
        self,
        patient_id: str,
        indication: str,
        treatment_criteria: list[TreatmentCriteria],
    ) -> RankingResult:
        """
        Rank treatments using ensemble of WeightedScoring + TOPSIS.

        Parameters
        ----------
        patient_id : str
            Patient identifier.
        indication : str
            Clinical indication.
        treatment_criteria : list[TreatmentCriteria]
            List of treatment criteria scores.

        Returns
        -------
        RankingResult
            Ensemble-ranked treatments.
        """
        ws_result   = self._weighted.rank(patient_id, indication, treatment_criteria)
        topsis_result = self._topsis.rank(patient_id, indication, treatment_criteria)

        # Build rank lookup per method
        ws_ranks = {r.treatment_name: r.rank for r in ws_result.ranked_treatments}
        tp_ranks = {r.treatment_name: r.rank for r in topsis_result.ranked_treatments}
        ws_scores = {r.treatment_name: r.composite_score for r in ws_result.ranked_treatments}
        tp_scores = {r.treatment_name: r.composite_score for r in topsis_result.ranked_treatments}

        # Average rank (Borda-style ensemble)
        all_names = list({r.treatment_name for r in ws_result.ranked_treatments})
        ensemble_scores = {}
        for name in all_names:
            ws_r = ws_ranks.get(name, len(all_names) + 1)
            tp_r = tp_ranks.get(name, len(all_names) + 1)
            avg_rank  = (ws_r + tp_r) / 2
            avg_score = (ws_scores.get(name, 0) + tp_scores.get(name, 0)) / 2
            ensemble_scores[name] = (avg_rank, avg_score)

        sorted_names = sorted(
            ensemble_scores.keys(),
            key=lambda n: ensemble_scores[n][0],
        )

        ranked = []
        for i, name in enumerate(sorted_names, 1):
            avg_rank, avg_score = ensemble_scores[name]
            confidence = (
                "high"   if avg_score >= 0.7 else
                "medium" if avg_score >= 0.45 else
                "low"
            )
            ranked.append(RankedTreatment(
                rank=i,
                treatment_name=name,
                composite_score=round(avg_score, 3),
                ranking_method="ensemble",
                criteria_scores={
                    "weighted_scoring_rank": ws_ranks.get(name, 0),
                    "topsis_rank":           tp_ranks.get(name, 0),
                    "weighted_score":        round(ws_scores.get(name, 0), 3),
                    "topsis_score":          round(tp_scores.get(name, 0), 3),
                },
                confidence=confidence,
                rationale=f"Ensemble: WS rank {ws_ranks.get(name,'?')}, TOPSIS rank {tp_ranks.get(name,'?')}",
            ))

        logger.info(
            "Ensemble ranked %d treatments for %s [%s]: top=%s",
            len(ranked), patient_id, indication,
            ranked[0].treatment_name if ranked else "none",
        )

        return RankingResult(
            patient_id=patient_id,
            indication=indication,
            method="ensemble",
            ranked_treatments=ranked,
            weights_used=self._weighted.weights,
            n_treatments_evaluated=len(ranked),
        )
