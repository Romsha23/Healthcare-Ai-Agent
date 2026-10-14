"""
tests/test_pdf_generator.py
───────────────────────────
Tests for the clinical summary PDF generator utility.
"""

from src.utils.pdf_generator import generate_clinical_summary_pdf


def test_generate_clinical_summary_pdf_basic():
    summary_text = (
        "### CLINICAL DISCHARGE SUMMARY\n\n"
        "Patient evaluated for Heart Failure and Hypertension.\n\n"
        "#### Recommendations\n"
        "1. Continue furosemide 40mg daily\n"
        "2. Follow up in clinic in 7 days\n"
    )
    entities = {
        "patient_id": "P-TEST-01",
        "patient_name": "Jane Doe",
        "age": "65",
        "gender": "Female",
        "conditions": ["Heart Failure", "Hypertension"],
        "medications": ["furosemide 40mg", "lisinopril 20mg"],
        "vitals": {"BP": "138/84 mmHg", "HR": "72 bpm"},
        "egfr": 45.0,
    }
    key_alerts = ["MEDICINE CLASH ALERT: Review ACEi and diuretic doses."]
    action_items = ["Schedule renal panel check in 10 days.", "Daily weight monitoring."]
    advices = {
        "diet": ["Low sodium diet (<2000mg/day).", "Fluid limit 1.5-2.0L."],
        "medications": ["Take diuretic in morning."],
        "warnings": ["Report sudden weight gain > 2-3 lbs."],
        "lifestyle": ["Daily morning weight log."],
    }

    pdf_bytes = generate_clinical_summary_pdf(
        summary_text=summary_text,
        entities=entities,
        key_alerts=key_alerts,
        action_items=action_items,
        advices=advices,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF magic byte signature %PDF-
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_clinical_summary_pdf_empty_or_minimal():
    pdf_bytes = generate_clinical_summary_pdf(
        summary_text="Routine checkup normal.",
        entities={"patient_id": "P002"},
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")
