"""
agents/summarizer_agent.py
───────────────────────────
Clinical document summarization agent.

Generates structured clinical summaries from:
    - Patient FHIR records (SBAR format)
    - Clinical study reports (executive summary)
    - Discharge summaries (transition of care)
    - Medication reconciliation reports
    - Radiology and pathology reports
    - Multi-visit longitudinal patient summaries

Follows clinical documentation standards:
    - SBAR (Situation Background Assessment Recommendation)
    - HL7 CDA (Clinical Document Architecture) structure
    - Joint Commission documentation requirements
    - CMS Transition of Care standards

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# ── Summary types ─────────────────────────────────────────────────────────────

SUMMARY_FORMATS = {
    "sbar":           "SBAR (Situation, Background, Assessment, Recommendation)",
    "discharge":      "Discharge Summary (Transition of Care)",
    "medication_rec": "Medication Reconciliation",
    "radiology":      "Radiology Report Summary",
    "pathology":      "Pathology Report Summary",
    "longitudinal":   "Longitudinal Patient Summary",
    "executive":      "Executive Clinical Summary",
    "handoff":        "Clinical Handoff (I-PASS format)",
}

# System prompts per summary type
SYSTEM_PROMPTS = {
    "sbar": """
You are a senior clinical nurse specialist creating an SBAR summary
for clinical handoff. Generate a precise, actionable SBAR:

SITUATION: Current clinical issue in 1-2 sentences
BACKGROUND: Relevant medical history, medications, allergies
ASSESSMENT: Clinical findings, lab values, diagnostic results
RECOMMENDATION: Specific action items, monitoring parameters, follow-up

Rules:
- Be concise — target 300 words maximum
- Flag URGENT items in CAPS
- Include specific values (lab results, vital signs, doses)
- End with clear, prioritized action items
- This is a SUPPORT tool — clinical judgment takes precedence
""",
    "discharge": """
You are a hospitalist physician creating a discharge summary
per CMS Transition of Care standards. Include:

1. ADMISSION DIAGNOSIS and DISCHARGE DIAGNOSIS
2. HOSPITAL COURSE — key events, procedures, clinical decisions
3. DISCHARGE CONDITION — vital signs, functional status
4. DISCHARGE MEDICATIONS — complete reconciled list with doses
5. PENDING RESULTS — labs/imaging awaiting final read
6. FOLLOW-UP — appointments, timeline, responsible providers
7. PATIENT INSTRUCTIONS — warning signs requiring return to ED

Be thorough but concise. Flag any pending items clearly.
""",
    "medication_rec": """
You are a clinical pharmacist performing medication reconciliation.
Generate a structured report:

1. PRE-ADMISSION MEDICATIONS — complete list
2. INPATIENT MEDICATIONS — added, changed, or held
3. DISCHARGE MEDICATIONS — final reconciled list
4. DISCREPANCIES — any unresolved differences with explanation
5. HIGH-ALERT MEDICATIONS — special instructions for each
6. PATIENT COUNSELING POINTS — key messages for patient education

Flag any ALLERGY CONFLICTS or DRUG INTERACTIONS clearly.
""",
    "handoff": """
You are generating an I-PASS clinical handoff summary:

I - ILLNESS SEVERITY: Stable / Watcher / Unstable
P - PATIENT SUMMARY: One-liner, active problems
A - ACTION LIST: To-do items with owner and timeline
S - SITUATION AWARENESS: Anticipated events and contingency plans
S - SYNTHESIS BY RECEIVER: [Space for receiver acknowledgment]

Be specific about overnight tasks and contingency plans.
""",
    "longitudinal": """
You are a clinical informaticist creating a longitudinal patient summary
for population health and care coordination. Include:

1. DEMOGRAPHICS and CARE TEAM
2. CHRONIC CONDITIONS — date of diagnosis, current status, control
3. MEDICATION HISTORY — current, past, reasons for changes
4. PREVENTIVE CARE — screenings, immunizations, gaps
5. HOSPITALIZATIONS — key admissions and outcomes
6. CARE GAPS — overdue screenings, uncontrolled conditions
7. SOCIAL DETERMINANTS — relevant SDOH factors

Flag care gaps and actionable items clearly.
""",
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ClinicalSummary:
    """A generated clinical summary."""
    patient_id: str
    summary_type: str
    summary_text: str
    word_count: int = 0
    key_alerts: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    model_used: str = ""

    def __post_init__(self):
        self.word_count = len(self.summary_text.split())

    def to_dict(self) -> dict:
        return {
            "patient_id":   self.patient_id,
            "summary_type": self.summary_type,
            "word_count":   self.word_count,
            "key_alerts":   self.key_alerts,
            "action_items": self.action_items,
            "summary_text": self.summary_text,
        }


# ── Summarizer Agent ──────────────────────────────────────────────────────────

class ClinicalSummarizerAgent:
    """
    Clinical document summarization agent using GPT-4o.

    Generates structured clinical summaries in multiple formats
    including SBAR, discharge summaries, medication reconciliation,
    I-PASS handoffs, and longitudinal patient summaries.

    Follows Joint Commission, CMS, and HL7 CDA documentation
    standards for clinical document quality.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key.
    model : str
        LLM model. Default 'gpt-4o'.
    max_tokens : int
        Maximum summary length in tokens. Default 1500.

    Examples
    --------
    >>> agent = ClinicalSummarizerAgent()
    >>> summary = agent.summarize(
    ...     clinical_text="Patient John Doe, 65M admitted with...",
    ...     patient_id="P001",
    ...     summary_type="sbar",
    ... )
    >>> print(summary.summary_text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 1500,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=."
            )
        self.model      = model
        self.max_tokens = max_tokens
        self._llm       = ChatOpenAI(
            model=model,
            temperature=0.1,
            openai_api_key=api_key,
            max_tokens=max_tokens,
        )
        logger.info("ClinicalSummarizerAgent initialized: model=%s", model)

    # ── Public API ────────────────────────────────────────────────────────────

    def summarize(
        self,
        clinical_text: str,
        patient_id: str,
        summary_type: str = "sbar",
        additional_context: Optional[str] = None,
    ) -> ClinicalSummary:
        """
        Generate a structured clinical summary.

        Parameters
        ----------
        clinical_text : str
            Clinical document text to summarize.
        patient_id : str
            Patient identifier.
        summary_type : str
            Summary format. One of: sbar, discharge, medication_rec,
            handoff, longitudinal, executive. Default 'sbar'.
        additional_context : str, optional
            Additional context to include (e.g. drug safety findings).

        Returns
        -------
        ClinicalSummary
        """
        if summary_type not in SUMMARY_FORMATS:
            logger.warning(
                "Unknown summary type '%s' — defaulting to sbar", summary_type
            )
            summary_type = "sbar"

        system_prompt = SYSTEM_PROMPTS.get(summary_type, SYSTEM_PROMPTS["sbar"])
        context_block = f"\nAdditional context:\n{additional_context}" if additional_context else ""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Patient ID: {patient_id}\n\n"
                f"Clinical information:\n{clinical_text}"
                f"{context_block}\n\n"
                f"Generate a {SUMMARY_FORMATS[summary_type]}."
            )),
        ]

        response = self._llm.invoke(messages)
        text = response.content.strip()

        alerts = self._extract_alerts(text)
        actions = self._extract_actions(text)

        summary = ClinicalSummary(
            patient_id=patient_id,
            summary_type=summary_type,
            summary_text=text,
            key_alerts=alerts,
            action_items=actions,
            model_used=self.model,
        )

        logger.info(
            "Summary generated [%s, %s]: %d words, %d alerts, %d actions",
            patient_id, summary_type,
            summary.word_count, len(alerts), len(actions),
        )
        return summary

    def batch_summarize(
        self,
        cases: list[dict],
        summary_type: str = "sbar",
    ) -> list[ClinicalSummary]:
        """
        Generate summaries for multiple patients.

        Parameters
        ----------
        cases : list[dict]
            List of dicts with 'patient_id' and 'clinical_text' keys.
        summary_type : str
            Summary format to use for all cases.

        Returns
        -------
        list[ClinicalSummary]
        """
        summaries = []
        for i, case in enumerate(cases, 1):
            logger.info(
                "Summarizing case %d/%d: %s",
                i, len(cases), case.get("patient_id", "unknown"),
            )
            summary = self.summarize(
                clinical_text=case.get("clinical_text", ""),
                patient_id=case.get("patient_id", f"patient_{i}"),
                summary_type=summary_type,
                additional_context=case.get("additional_context"),
            )
            summaries.append(summary)
        return summaries

    def extract_key_information(
        self,
        clinical_text: str,
        information_type: str = "medications",
    ) -> list[str]:
        """
        Extract specific clinical information from unstructured text.

        Parameters
        ----------
        clinical_text : str
            Clinical document text.
        information_type : str
            Type of information to extract:
            'medications', 'allergies', 'diagnoses',
            'procedures', 'lab_values', 'vital_signs'

        Returns
        -------
        list[str]
            Extracted items as a list of strings.
        """
        extraction_prompts = {
            "medications":  "Extract ALL medications with dose and frequency. One per line.",
            "allergies":    "Extract ALL documented allergies and adverse drug reactions. One per line.",
            "diagnoses":    "Extract ALL diagnoses and conditions (active and historical). One per line.",
            "procedures":   "Extract ALL procedures performed. One per line.",
            "lab_values":   "Extract ALL lab values with date and reference range status. One per line.",
            "vital_signs":  "Extract ALL vital signs with timestamps. One per line.",
        }

        prompt = extraction_prompts.get(
            information_type,
            f"Extract all {information_type} information. One item per line."
        )

        messages = [
            SystemMessage(content=(
                "You are a clinical information extraction specialist. "
                "Extract the requested information precisely. "
                "Return ONLY the extracted items, one per line. "
                "If an item is not present, return 'Not documented'."
            )),
            HumanMessage(content=f"{prompt}\n\nClinical text:\n{clinical_text}"),
        ]

        response = self._llm.invoke(messages)
        items = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and line.strip() != "-"
        ]

        logger.info(
            "Extracted %d %s items", len(items), information_type
        )
        return items

    def compare_summaries(
        self,
        summary1: ClinicalSummary,
        summary2: ClinicalSummary,
    ) -> dict:
        """
        Compare two clinical summaries to identify changes over time.

        Useful for tracking patient progress across admissions
        or comparing pre/post-treatment summaries.

        Parameters
        ----------
        summary1 : ClinicalSummary
            Earlier summary (baseline).
        summary2 : ClinicalSummary
            Later summary (follow-up).

        Returns
        -------
        dict
            Changes identified between the two summaries.
        """
        messages = [
            SystemMessage(content=(
                "You are a clinical data analyst comparing two clinical summaries "
                "for the same patient at different time points. Identify: "
                "1. New conditions or diagnoses, "
                "2. Medication changes (added, stopped, dose changes), "
                "3. Clinical status changes, "
                "4. Resolved vs ongoing problems. "
                "Be specific and clinically precise."
            )),
            HumanMessage(content=(
                f"EARLIER SUMMARY:\n{summary1.summary_text}\n\n"
                f"LATER SUMMARY:\n{summary2.summary_text}\n\n"
                "Identify key clinical changes between these two time points."
            )),
        ]

        response = self._llm.invoke(messages)
        return {
            "patient_id":   summary1.patient_id,
            "changes":      response.content.strip(),
            "new_alerts":   [
                a for a in summary2.key_alerts
                if a not in summary1.key_alerts
            ],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_alerts(text: str) -> list[str]:
        """Extract urgent alert items from summary text."""
        alerts = []
        alert_keywords = [
            "URGENT", "CRITICAL", "ALERT", "ALLERGY", "CONTRAINDICATED",
            "MAJOR INTERACTION", "HOLD", "STAT", "IMMEDIATE",
        ]
        for line in text.split("\n"):
            line_upper = line.upper()
            if any(kw in line_upper for kw in alert_keywords):
                alerts.append(line.strip())
        return alerts[:10]

    @staticmethod
    def _extract_actions(text: str) -> list[str]:
        """Extract action items from summary text."""
        actions = []
        action_keywords = [
            "RECOMMEND", "ORDER", "MONITOR", "FOLLOW-UP", "REFER",
            "ADJUST", "START", "STOP", "HOLD", "CHECK", "OBTAIN",
            "SCHEDULE", "NOTIFY", "REPEAT",
        ]
        for line in text.split("\n"):
            line_upper = line.upper()
            if any(kw in line_upper for kw in action_keywords):
                clean = line.strip().lstrip("•-* ")
                if len(clean) > 10:
                    actions.append(clean)
        return actions[:10]
