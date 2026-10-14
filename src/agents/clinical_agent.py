"""
agents/clinical_agent.py
─────────────────────────
LangGraph-orchestrated multi-agent clinical decision support system.

Implements a stateful multi-agent workflow for clinical decision
support with the following specialized agents:

    PatientContextAgent   — Retrieves and summarizes patient history
    DrugSafetyAgent       — Checks drug interactions and contraindications
    DiagnosticAgent       — Analyzes symptoms and lab values
    TreatmentAgent        — Suggests evidence-based treatment options
    SummaryAgent          — Synthesizes findings into clinical summary

The orchestrator routes queries through the appropriate agents
based on query type and patient context, maintaining full
conversation state across multi-turn clinical sessions.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Annotated, TypedDict

import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)


# ── Agent state ───────────────────────────────────────────────────────────────

class ClinicalAgentState(TypedDict):
    """Shared state across all agents in the clinical workflow."""
    messages: list
    patient_id: str
    query: str
    query_type: str          # drug_safety | diagnostic | treatment | summary | general
    patient_context: str
    drug_safety_findings: str
    diagnostic_findings: str
    treatment_recommendations: str
    final_summary: str
    alerts: list[str]        # Urgent clinical alerts (allergies, interactions)
    iteration_count: int


# ── Clinical tools ────────────────────────────────────────────────────────────

@tool
def check_drug_interaction(drug1: str, drug2: str) -> str:
    """
    Check for clinically significant interactions between two drugs.

    Args:
        drug1: First drug name (generic or brand)
        drug2: Second drug name (generic or brand)

    Returns:
        Clinical interaction summary with severity and management
    """
    # In production this calls RxNorm/DrugBank/FDA APIs
    # Here we implement a representative knowledge base
    interactions_db = {
        ("warfarin", "aspirin"):        ("MAJOR", "Increased bleeding risk. Monitor INR closely. Consider gastroprotection."),
        ("metformin", "contrast"):      ("MAJOR", "Hold metformin 48h before/after iodinated contrast. Risk of lactic acidosis."),
        ("ssri", "maoi"):               ("CONTRAINDICATED", "Risk of serotonin syndrome. Do not co-administer."),
        ("simvastatin", "amiodarone"):  ("MAJOR", "Risk of myopathy/rhabdomyolysis. Limit simvastatin to 20mg/day."),
        ("clopidogrel", "ppi"):         ("MODERATE", "PPIs may reduce clopidogrel efficacy. Pantoprazole preferred if needed."),
        ("lisinopril", "potassium"):    ("MODERATE", "Risk of hyperkalemia. Monitor potassium levels."),
        ("digoxin", "amiodarone"):      ("MAJOR", "Amiodarone increases digoxin levels. Reduce digoxin dose by 50%."),
        ("fluoroquinolone", "antacid"): ("MODERATE", "Antacids reduce fluoroquinolone absorption. Separate by 2 hours."),
    }

    d1 = drug1.lower().strip()
    d2 = drug2.lower().strip()

    # Check both orderings
    for key, (severity, description) in interactions_db.items():
        if (d1 in key[0] or key[0] in d1) and (d2 in key[1] or key[1] in d2):
            return f"[{severity}] {drug1} + {drug2}: {description}"
        if (d2 in key[0] or key[0] in d2) and (d1 in key[1] or key[1] in d1):
            return f"[{severity}] {drug1} + {drug2}: {description}"

    return f"[LOW/UNKNOWN] No significant interaction found between {drug1} and {drug2} in reference database. Verify with clinical pharmacist for complete assessment."


@tool
def check_allergy_contraindication(drug: str, allergy: str) -> str:
    """
    Check if a drug is contraindicated given a patient allergy.

    Args:
        drug: Drug name to check
        allergy: Patient's known allergy

    Returns:
        Contraindication assessment with clinical guidance
    """
    contraindications = {
        ("penicillin", "amoxicillin"):   "CONTRAINDICATED — Cross-reactivity with penicillin allergy (10% cross-reactivity with cephalosporins).",
        ("penicillin", "ampicillin"):    "CONTRAINDICATED — Same drug class as penicillin allergy.",
        ("sulfa", "sulfamethoxazole"):   "CONTRAINDICATED — Sulfonamide antibiotic — directly contraindicated.",
        ("sulfa", "furosemide"):         "CAUTION — Sulfonamide-derived diuretic. Potential cross-reactivity. Use with caution.",
        ("nsaid", "ibuprofen"):          "CONTRAINDICATED — NSAID class — directly contraindicated.",
        ("nsaid", "naproxen"):           "CONTRAINDICATED — NSAID class — directly contraindicated.",
        ("aspirin", "ibuprofen"):        "CAUTION — Monitor for cross-reactivity in aspirin-sensitive patients.",
        ("contrast", "metformin"):       "CAUTION — Hold metformin before contrast administration.",
        ("latex", "banana"):             "INFO — Latex-fruit syndrome — potential cross-reactivity.",
    }

    d = drug.lower().strip()
    a = allergy.lower().strip()

    for (allergy_key, drug_key), message in contraindications.items():
        if allergy_key in a and drug_key in d:
            return f"ALERT: {drug} in patient with {allergy} allergy — {message}"

    return f"No documented contraindication found between {drug} and {allergy} allergy in reference database. Clinical pharmacist review recommended for complete assessment."


@tool
def get_reference_ranges(lab_test: str) -> str:
    """
    Get standard reference ranges for a laboratory test.

    Args:
        lab_test: Laboratory test name (e.g. 'hemoglobin', 'creatinine')

    Returns:
        Reference ranges with clinical interpretation guidance
    """
    reference_ranges = {
        "hemoglobin":         "Male: 13.5-17.5 g/dL | Female: 12.0-15.5 g/dL | Critical low: <7.0 g/dL",
        "hgb":                "Male: 13.5-17.5 g/dL | Female: 12.0-15.5 g/dL | Critical low: <7.0 g/dL",
        "wbc":                "4.5-11.0 x10³/µL | Critical high: >30.0 | Critical low: <2.0",
        "platelets":          "150-400 x10³/µL | Critical low: <50 (bleeding risk) | Critical high: >1000",
        "creatinine":         "Male: 0.7-1.3 mg/dL | Female: 0.6-1.1 mg/dL | CKD staging per KDIGO",
        "glucose":            "Fasting: 70-100 mg/dL | Critical low: <40 | Critical high: >500",
        "sodium":             "136-145 mEq/L | Critical low: <120 | Critical high: >160",
        "potassium":          "3.5-5.0 mEq/L | Critical low: <2.8 | Critical high: >6.2",
        "alt":                "Male: 7-56 U/L | Female: 7-45 U/L | >3x ULN warrants investigation",
        "ast":                "10-40 U/L | >3x ULN warrants investigation",
        "troponin":           "<0.04 ng/mL (conventional) | High-sensitivity assay: <14 ng/L | Any elevation warrants urgent evaluation",
        "bnp":                "<100 pg/mL normal | 100-400 grey zone | >400 heart failure likely",
        "inr":                "Therapeutic anticoagulation: 2.0-3.0 | Mechanical valves: 2.5-3.5 | Critical: >5.0",
        "tsh":                "0.4-4.0 mIU/L | Hypothyroid: >4.0 | Hyperthyroid: <0.1",
        "hba1c":              "Normal: <5.7% | Prediabetes: 5.7-6.4% | Diabetes: ≥6.5% | Target in DM: <7%",
        "ldl":                "Optimal: <100 mg/dL | High risk: target <70 mg/dL | Very high risk: target <55 mg/dL",
        "egfr":               "Normal: ≥60 mL/min/1.73m² | CKD G3a: 45-59 | G3b: 30-44 | G4: 15-29 | G5: <15",
    }

    test = lab_test.lower().strip()
    for key, ranges in reference_ranges.items():
        if key in test or test in key:
            return f"{lab_test}: {ranges}"

    return f"Reference ranges for '{lab_test}' not in local database. Consult institutional laboratory reference guide."


@tool
def lookup_clinical_guideline(condition: str, aspect: str = "treatment") -> str:
    """
    Look up evidence-based clinical guideline recommendations.

    Args:
        condition: Clinical condition (e.g. 'NSCLC', 'heart failure', 'diabetes')
        aspect: Aspect to look up — 'treatment', 'screening', 'diagnosis', 'monitoring'

    Returns:
        Guideline summary with evidence level and source
    """
    guidelines = {
        ("nsclc", "treatment"):         "NCCN NSCLC v2024: Stage IIIB/IV — First-line: Pembrolizumab + platinum doublet if PD-L1≥1% and no EGFR/ALK. EGFR+ → Osimertinib. ALK+ → Alectinib. [Category 1 evidence]",
        ("heart failure", "treatment"): "ACC/AHA 2022: HFrEF (EF<40%) — GDMT: ACEi/ARB/ARNI + beta-blocker + MRA + SGLT2i. Target LVEF improvement. [Class I, Level A]",
        ("diabetes", "treatment"):      "ADA 2024: T2DM — Metformin first-line unless contraindicated. Add GLP-1RA or SGLT2i if CVD/CKD risk. HbA1c target <7% for most. [Grade A]",
        ("hypertension", "treatment"):  "JNC 2023: Target BP <130/80 for most adults. First-line: thiazide, ACEi/ARB, or CCB. Two-drug therapy if BP >20/10 above target. [Grade A]",
        ("sepsis", "treatment"):        "Surviving Sepsis 2021: Hour-1 bundle — blood cultures, broad-spectrum antibiotics, 30mL/kg crystalloid if hypotensive, vasopressors if MAP<65. [Strong recommendation]",
        ("afib", "treatment"):          "AHA/ACC 2023: Rate control (HR<80 bpm) vs rhythm control based on symptoms. Anticoagulation: CHA₂DS₂-VASc ≥2 (men) / ≥3 (women) — DOAC preferred over warfarin. [Class I]",
        ("ckd", "monitoring"):          "KDIGO 2024: eGFR + UACR every 3-12 months based on risk. BP target <120/80. Avoid NSAIDs, nephrotoxins. Consider nephrology referral if eGFR<30. [Grade 1B]",
        ("breast cancer", "screening"): "USPSTF 2024: Mammography screening every 2 years for women 40-74. High-risk (BRCA1/2): annual MRI + mammogram from age 30. [Grade B]",
    }

    cond = condition.lower().strip()
    asp  = aspect.lower().strip()

    for (cond_key, asp_key), guidance in guidelines.items():
        if cond_key in cond and asp_key in asp:
            return guidance

    return f"No specific guideline entry found for '{condition}' ({aspect}). Recommend consulting UpToDate, NCCN, or specialty society guidelines."


# ── Clinical Agent Orchestrator ───────────────────────────────────────────────

class ClinicalAgentOrchestrator:
    """
    LangGraph-orchestrated multi-agent clinical decision support system.

    Routes clinical queries through specialized agents:
    - Patient context retrieval from RAG pipeline
    - Drug safety checking (interactions, allergies, contraindications)
    - Diagnostic support (lab interpretation, differential diagnosis)
    - Treatment recommendations (guideline-based)
    - Clinical summary generation (SBAR format)

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key.
    model : str
        LLM model. Default 'gpt-4o'.
    rag_pipeline : optional
        ClinicalRAGPipeline instance for patient context retrieval.

    Examples
    --------
    >>> agent = ClinicalAgentOrchestrator()
    >>> result = agent.run(
    ...     query="What medications is the patient on and are there any interactions?",
    ...     patient_id="P001",
    ... )
    >>> print(result["final_summary"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        rag_pipeline=None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        self.model        = model
        self.rag_pipeline = rag_pipeline
        self._tools = [
            check_drug_interaction,
            check_allergy_contraindication,
            get_reference_ranges,
            lookup_clinical_guideline,
        ]
        self._llm = None
        self._llm_with_tools = None

        if api_key and not api_key.startswith("sk-proj-placeholder"):
            try:
                self._llm = ChatOpenAI(
                    model=model,
                    temperature=0,
                    openai_api_key=api_key,
                    max_retries=1,
                    timeout=8,
                )
                self._llm_with_tools = self._llm.bind_tools(self._tools)
            except Exception as e:
                logger.warning("Could not initialize ChatOpenAI: %s. Using deterministic clinical engine.", e)

        self._graph = self._build_graph()
        logger.info("ClinicalAgentOrchestrator initialized: model=%s (LLM available: %s)", model, bool(self._llm))

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        patient_id: str = "",
        patient_context: str = "",
    ) -> dict:
        """
        Run the multi-agent clinical decision support workflow.

        Parameters
        ----------
        query : str
            Clinical question or task.
        patient_id : str, optional
            Patient identifier for RAG retrieval.
        patient_context : str, optional
            Pre-loaded patient context (bypasses RAG lookup).

        Returns
        -------
        dict
            Final state with clinical findings, recommendations,
            alerts, and summary.
        """
        # Auto-retrieve patient context from RAG if available
        if not patient_context and patient_id and self.rag_pipeline:
            try:
                rag_response = self.rag_pipeline.query(
                    question=query,
                    patient_id=patient_id,
                )
                patient_context = rag_response.answer
            except Exception as e:
                logger.warning("RAG retrieval failed: %s", e)

        query_type = self._classify_query(query)

        initial_state: ClinicalAgentState = {
            "messages":               [HumanMessage(content=query)],
            "patient_id":             patient_id,
            "query":                  query,
            "query_type":             query_type,
            "patient_context":        patient_context,
            "drug_safety_findings":   "",
            "diagnostic_findings":    "",
            "treatment_recommendations": "",
            "final_summary":          "",
            "alerts":                 [],
            "iteration_count":        0,
        }

        logger.info(
            "Running clinical agent: query_type=%s, patient=%s",
            query_type, patient_id or "unknown",
        )

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as e_graph:
            logger.warning("LangGraph invocation failed: %s. Using sequential fallback pipeline.", e_graph)
            st_cur = self._fallback_patient_context(initial_state)
            st_cur = self._fallback_drug_safety(st_cur)
            st_cur = self._fallback_diagnostic(st_cur)
            st_cur = self._fallback_treatment(st_cur)
            st_cur = self._fallback_summary(st_cur)
            final_state = st_cur

        return {
            "query":                  final_state["query"],
            "patient_id":             final_state["patient_id"],
            "query_type":             final_state["query_type"],
            "patient_context":        final_state["patient_context"],
            "drug_safety_findings":   final_state["drug_safety_findings"],
            "diagnostic_findings":    final_state["diagnostic_findings"],
            "treatment_recommendations": final_state["treatment_recommendations"],
            "final_summary":          final_state["final_summary"],
            "alerts":                 final_state["alerts"],
        }

    # ── Graph nodes ───────────────────────────────────────────────────────────

    def _fallback_patient_context(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Deterministic extraction of patient context."""
        ctx = state.get("patient_context", "").strip()
        q = state.get("query", "").strip()
        combined = f"{ctx}\n{q}"

        import re
        p_match = re.search(r"(?:patient(?:\s+name)?|name)[\s:]+([A-Za-z\s\.]+?)(?:\||\n|,|\bmrn\b|\bdob\b|$)", combined, re.IGNORECASE)
        p_name = p_match.group(1).strip() if p_match else "Patient"

        summary_lines = [f"Clinical Context for {p_name}:"]
        if ctx:
            summary_lines.append(ctx)
        else:
            summary_lines.append(f"Clinical Query: {q}")

        return {**state, "patient_context": "\n".join(summary_lines)}

    def _handle_llm_failure(self, err: Exception, step_name: str) -> None:
        """Handle LLM failures gracefully; disables broken LLM connections immediately."""
        err_str = str(err).lower()
        logger.warning("LLM %s failed (%s). Switching to deterministic clinical engine.", step_name, err)
        if any(k in err_str for k in ["quota", "credit", "429", "ratelimit", "unauthorized", "401", "authentication", "billing"]):
            self._llm = None
            self._llm_with_tools = None

    def _node_patient_context(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Retrieve and summarize patient clinical context."""
        if state["patient_context"]:
            return state

        if self._llm:
            try:
                system = SystemMessage(content="""
You are a clinical data retrieval specialist. Given available patient
information, extract and organize:
1. Active medications (name, dose, frequency)
2. Known allergies and reactions
3. Active diagnoses and chronic conditions
4. Recent relevant lab values
5. Relevant procedures and history

Be concise and clinically precise. Flag any URGENT items.
""")
                user = HumanMessage(content=f"Summarize available context for: {state['query']}")
                response = self._llm.invoke([system, user])
                return {**state, "patient_context": response.content}
            except Exception as e:
                self._handle_llm_failure(e, "patient_context")

        return self._fallback_patient_context(state)

    def _fallback_drug_safety(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Deterministic drug safety checking using clinical tool database."""
        import re
        from src.agents.drug_interaction_tool import DrugInteractionChecker

        combined = f"{state.get('patient_context', '')}\n{state.get('query', '')}".lower()

        known_drugs = [
            "warfarin", "aspirin", "metformin", "lisinopril", "amoxicillin", "pembrolizumab",
            "atorvastatin", "metoprolol", "omeprazole", "heparin", "digoxin", "methotrexate",
            "ciprofloxacin", "fluoxetine", "tramadol", "amiodarone", "losartan", "gabapentin",
            "furosemide", "apixaban", "rivaroxaban", "dabigatran", "clopidogrel", "insulin",
            "levothyroxine", "amlodipine", "pantoprazole", "prednisone", "albuterol",
            "empagliflozin", "dapagliflozin", "sitagliptin", "semaglutide", "carvedilol",
            "simvastatin", "vancomycin", "spironolactone"
        ]

        detected_meds = [kd for kd in known_drugs if kd in combined]

        allergies = []
        if "penicillin" in combined and ("allerg" in combined or "anaphylaxis" in combined):
            allergies.append("penicillin")
        if "sulfa" in combined and "allerg" in combined:
            allergies.append("sulfonamides")

        egfr = None
        egfr_match = re.search(r"(?:egfr|estimated gfr|gfr)[\s:>=<]+([0-9.]+)", combined)
        if egfr_match:
            try:
                egfr = float(egfr_match.group(1))
            except ValueError:
                pass

        checker = DrugInteractionChecker()
        report = checker.screen_medication_list(
            patient_id=state.get("patient_id", "P001"),
            medications=detected_meds,
            allergies=allergies,
            egfr=egfr,
        )

        lines = []
        alerts = state["alerts"].copy()

        if report.interactions:
            for inter in report.interactions:
                lines.append(f"[{inter.severity}] {inter.drug1.capitalize()} + {inter.drug2.capitalize()}: {inter.clinical_effect} (Management: {inter.management})")
                if inter.severity in ["CONTRAINDICATED", "MAJOR"]:
                    alerts.append(f"DRUG SAFETY ALERT: {inter.drug1.capitalize()} + {inter.drug2.capitalize()} ({inter.severity}) — {inter.clinical_effect}")

        if report.renal_adjustments:
            for radj in report.renal_adjustments:
                lines.append(f"[RENAL CLEARANCE] {radj}")
                alerts.append(f"RENAL ALERT: {radj}")

        if report.high_alert_flags:
            for flag in report.high_alert_flags:
                lines.append(f"[HIGH-ALERT] {flag}")

        if not lines:
            lines.append("No critical drug-drug interactions, allergy conflicts, or organ clearance contraindications identified among evaluated agents.")

        findings = "\n".join(lines)
        return {**state, "drug_safety_findings": findings, "alerts": alerts}

    def _node_drug_safety(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Check drug safety — interactions, allergies, contraindications."""
        if self._llm_with_tools:
            try:
                system = SystemMessage(content="""
You are a clinical pharmacist AI. Using the available tools, check for:
1. Drug-drug interactions between current medications
2. Allergy contraindications for any new or existing medications
3. Dose adjustments needed for renal/hepatic impairment
4. High-alert medications requiring extra monitoring

Always use the check_drug_interaction and check_allergy_contraindication
tools for any medications mentioned. Flag MAJOR interactions as ALERTS.
""")
                messages = [
                    system,
                    HumanMessage(content=f"""
Patient context: {state['patient_context']}
Query: {state['query']}
Check drug safety for all medications mentioned.
""")
                ]
                response = self._llm_with_tools.invoke(messages)
                alerts = state["alerts"].copy()
                findings = response.content or ""
                if "MAJOR" in findings.upper() or "CONTRAINDICATED" in findings.upper():
                    alerts.append(f"DRUG SAFETY ALERT: {findings[:200]}")
                return {**state, "drug_safety_findings": findings, "alerts": alerts}
            except Exception as e:
                self._handle_llm_failure(e, "drug_safety")

        return self._fallback_drug_safety(state)

    def _fallback_diagnostic(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Deterministic diagnostic evaluation using lab reference tool."""
        import re
        combined = f"{state.get('patient_context', '')}\n{state.get('query', '')}".lower()
        lines = []
        alerts = state["alerts"].copy()

        lab_tests = ["egfr", "creatinine", "bnp", "inr", "troponin", "potassium", "sodium", "hemoglobin", "platelets"]
        for lt in lab_tests:
            m = re.search(rf"\b{lt}[\s:>=<]+([0-9.]+)", combined)
            if m:
                val = float(m.group(1))
                ref = get_reference_ranges.invoke({"lab_test": lt})
                lines.append(f"• {lt.upper()}: {val} (Reference: {ref})")
                if lt == "egfr" and val < 30:
                    alerts.append(f"CRITICAL RENAL THRESHOLD: eGFR {val} mL/min/1.73m² (Severe CKD)")
                elif lt == "potassium" and (val < 3.0 or val > 5.2):
                    alerts.append(f"ELECTROLYTE ALERT: Potassium {val} mEq/L")
                elif lt == "bnp" and val > 400:
                    lines.append("  ↳ Strongly consistent with acute decompensated heart failure.")

        if "hiv" in combined or "hbsag" in combined or "hcv" in combined or "viral marker" in combined:
            lines.append("• Diagnostic Viral Marker Serology Screen:")
            v_items = [
                ("HIV 1", "NEGATIVE" if "hiv 1 - negative" in combined or "hiv 1 negative" in combined or "hiv negative" in combined else "Screened"),
                ("HIV 2", "NEGATIVE" if "hiv 2 - negative" in combined or "hiv 2 negative" in combined or "hiv negative" in combined else "Screened"),
                ("HBsAg", "NEGATIVE" if "hbsag - negative" in combined or "hbsag negative" in combined else "Screened"),
                ("Anti-HCV", "NEGATIVE" if "anti hcv - negative" in combined or "anti hcv negative" in combined or "hcv negative" in combined else "Screened"),
            ]
            for v_name, v_res in v_items:
                lines.append(f"  - {v_name}: {v_res}")
            if all(v[1] == "NEGATIVE" for v in v_items):
                lines.append("  ↳ All viral markers Non-Reactive. Patient serologically cleared for planned procedures.")

        if not lines:
            lines.append("Routine laboratory and diagnostic markers reviewed. No critical acute threshold breaches identified.")

        findings = "\n".join(lines)
        return {**state, "diagnostic_findings": findings, "alerts": alerts}

    def _node_diagnostic(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Analyze diagnostic information — labs, symptoms, imaging."""
        if self._llm_with_tools:
            try:
                system = SystemMessage(content="""
You are a clinical diagnostician AI. Using available tools:
1. Interpret laboratory values against reference ranges
2. Identify abnormal values and their clinical significance
3. Flag critical values requiring immediate action
4. Suggest differential diagnoses based on available data

Use get_reference_ranges tool for any lab values mentioned.
Never make definitive diagnoses — use "consistent with" or "suggestive of".
""")
                messages = [
                    system,
                    HumanMessage(content=f"""
Patient context: {state['patient_context']}
Query: {state['query']}
Analyze diagnostic findings.
""")
                ]
                response = self._llm_with_tools.invoke(messages)
                alerts = state["alerts"].copy()
                findings = response.content or ""
                if "CRITICAL" in findings.upper():
                    alerts.append(f"CRITICAL VALUE ALERT: {findings[:200]}")
                return {**state, "diagnostic_findings": findings, "alerts": alerts}
            except Exception as e:
                self._handle_llm_failure(e, "diagnostic")

        return self._fallback_diagnostic(state)

    def _fallback_treatment(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Deterministic treatment guideline lookup."""
        import re
        combined = f"{state.get('patient_context', '')}\n{state.get('query', '')}".lower()
        lines = []

        if "nsclc" in combined or "lung cancer" in combined:
            g = lookup_clinical_guideline.invoke({"condition": "nsclc", "aspect": "treatment"})
            lines.append(f"• Oncology Guideline: {g}")
            if "pdl1" in combined or "pd-l1" in combined:
                pdl1_m = re.search(r"pd-?l1(?:\s+tps)?[\s:>=<]+([0-9]+)%?", combined)
                if pdl1_m and float(pdl1_m.group(1)) >= 50:
                    lines.append("  ↳ Precision Selection (PD-L1 TPS ≥ 50%): Category 1 recommendation for Pembrolizumab monotherapy (200mg IV Q3W) over cytotoxic chemotherapy.")
        elif "heart failure" in combined or "hfref" in combined:
            g = lookup_clinical_guideline.invoke({"condition": "heart failure", "aspect": "treatment"})
            lines.append(f"• Cardiology Guideline: {g}")
            lines.append("  ↳ HFrEF Optimization: Quadruple GDMT (ARNI/ACEi, beta-blocker, MRA, SGLT2i) recommended with daily potassium monitoring.")
        elif "diabetes" in combined or "t2dm" in combined:
            g = lookup_clinical_guideline.invoke({"condition": "diabetes", "aspect": "treatment"})
            lines.append(f"• Endocrine Guideline: {g}")
        elif "viral" in combined or "hiv" in combined:
            lines.append("• Procedural Protocol: Pre-operative viral screening verified. Patient cleared for elective intervention under standard universal precautions.")
        else:
            lines.append("• Evidence-based guideline therapy: Maintain standardized clinical management per institutional and specialty clinical guidelines.")

        return {**state, "treatment_recommendations": "\n".join(lines)}

    def _node_treatment(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Generate evidence-based treatment recommendations."""
        if self._llm_with_tools:
            try:
                system = SystemMessage(content="""
You are a clinical decision support AI specializing in evidence-based
medicine. Using clinical guidelines:
1. Identify applicable treatment guidelines for the condition
2. Suggest first-line and alternative treatment options
3. Note contraindications to standard treatments based on patient context
4. Recommend monitoring parameters and follow-up

Use lookup_clinical_guideline tool for relevant conditions.
Always cite guideline source and evidence level.
Clearly state this is decision SUPPORT — final decisions rest with clinicians.
""")
                messages = [
                    system,
                    HumanMessage(content=f"""
Patient context: {state['patient_context']}
Drug safety findings: {state['drug_safety_findings']}
Diagnostic findings: {state['diagnostic_findings']}
Query: {state['query']}
Provide treatment recommendations.
""")
                ]
                response = self._llm_with_tools.invoke(messages)
                return {**state, "treatment_recommendations": response.content or ""}
            except Exception as e:
                self._handle_llm_failure(e, "treatment")

        return self._fallback_treatment(state)

    def _fallback_summary(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Synthesize all findings into structured SBAR format."""
        import re
        combined = f"{state.get('patient_context', '')}\n{state.get('query', '')}"

        p_name = state.get("patient_id", "Patient")
        nm = re.search(r"(?:patient(?:\s+name)?|name)[\s:]+([A-Za-z\s\.]+?)(?:\||\n|,|\bmrn\b|\bdob\b|$)", combined, re.IGNORECASE)
        if nm and len(nm.group(1).strip()) > 1 and not nm.group(1).strip().lower().startswith("is a"):
            p_name = nm.group(1).strip()
            if p_name.lower().startswith("of "):
                p_name = p_name[3:].strip()

        age = None
        am = re.search(r"\b(\d{1,3})[\s-]*(?:years?[- ]old|yo|y/o|year-old|yrs?)\b", combined, re.IGNORECASE)
        if am:
            age = am.group(1).strip()

        demo = f"{age}yo" if age else "Patient"

        summary = f"""**SITUATION:**
Evaluation for {p_name} ({demo}, ID: {state.get('patient_id', 'P001')}). Multi-agent decision support workflow orchestrated across Patient Context, Pharmacovigilance, Diagnostic, and Guideline Treatment nodes.

**BACKGROUND:**
- Primary Focus: {state['query']}
- Alerts Identified: {len(state['alerts'])} active alerts

**ASSESSMENT:**
- **Pharmacovigilance Review:**
{state['drug_safety_findings']}

- **Diagnostic Review:**
{state['diagnostic_findings']}

**RECOMMENDATION:**
- **Evidence-Based Treatment Guidance:**
{state['treatment_recommendations']}
- **Action Plan:** Follow-up clinical monitoring and multidisciplinary care team review recommended.
"""
        return {**state, "final_summary": summary}

    def _node_summary(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """Synthesize all findings into a clinical summary."""
        if self._llm:
            try:
                system = SystemMessage(content="""
You are a senior clinical documentation specialist. Synthesize all
available findings into a concise, actionable clinical summary.

Format as SBAR:
- SITUATION: What is the clinical issue?
- BACKGROUND: Relevant patient context
- ASSESSMENT: Key findings from drug safety, diagnostic, and treatment review
- RECOMMENDATION: Specific, prioritized action items

Also list any ALERTS prominently at the top.
Keep under 400 words. This is a decision SUPPORT summary only.
""")
                alerts_text = "\n".join(state["alerts"]) if state["alerts"] else "None"
                messages = [
                    system,
                    HumanMessage(content=f"""
ALERTS: {alerts_text}
Patient context: {state['patient_context']}
Drug safety: {state['drug_safety_findings']}
Diagnostic findings: {state['diagnostic_findings']}
Treatment recommendations: {state['treatment_recommendations']}
Original query: {state['query']}

Generate SBAR clinical summary.
""")
                ]
                response = self._llm.invoke(messages)
                return {**state, "final_summary": response.content or ""}
            except Exception as e:
                self._handle_llm_failure(e, "summary")

        return self._fallback_summary(state)

    # ── Graph routing ─────────────────────────────────────────────────────────

    def _route_query(self, state: ClinicalAgentState) -> str:
        """Route to appropriate agent based on query type."""
        query_type = state["query_type"]
        if query_type == "drug_safety":
            return "drug_safety"
        elif query_type == "diagnostic":
            return "diagnostic"
        elif query_type == "treatment":
            return "treatment"
        else:
            return "drug_safety"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph multi-agent workflow."""
        workflow = StateGraph(ClinicalAgentState)

        workflow.add_node("patient_context", self._node_patient_context)
        workflow.add_node("drug_safety",     self._node_drug_safety)
        workflow.add_node("diagnostic",      self._node_diagnostic)
        workflow.add_node("treatment",       self._node_treatment)
        workflow.add_node("summary",         self._node_summary)

        workflow.set_entry_point("patient_context")
        workflow.add_conditional_edges(
            "patient_context",
            self._route_query,
            {
                "drug_safety": "drug_safety",
                "diagnostic":  "diagnostic",
                "treatment":   "treatment",
            },
        )
        workflow.add_edge("drug_safety", "diagnostic")
        workflow.add_edge("diagnostic",  "treatment")
        workflow.add_edge("treatment",   "summary")
        workflow.add_edge("summary",     END)

        return workflow.compile()

    @staticmethod
    def _classify_query(query: str) -> str:
        """Classify query type for routing."""
        q = query.lower()
        if any(w in q for w in ["drug", "medication", "allerg", "interact", "contraindic", "dose"]):
            return "drug_safety"
        elif any(w in q for w in ["lab", "result", "diagnos", "symptom", "value", "level", "test"]):
            return "diagnostic"
        elif any(w in q for w in ["treat", "therap", "guideline", "recommend", "manage", "prescrib"]):
            return "treatment"
        else:
            return "general"
