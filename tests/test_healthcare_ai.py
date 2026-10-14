

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key-123")
os.environ.setdefault("API_KEYS", "dev-key-123")


# ── FHIR fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_fhir_patient() -> dict:
    return {
        "resourceType": "Patient",
        "id": "P001",
        "name": [{"given": ["John"], "family": "Doe"}],
        "birthDate": "1958-03-15",
        "gender": "male",
        "identifier": [{"value": "MRN-001", "system": "urn:hospital:mrn"}],
    }

@pytest.fixture
def sample_fhir_condition() -> dict:
    return {
        "resourceType": "Condition",
        "id": "C001",
        "code": {"coding": [{"display": "Non-Small Cell Lung Cancer", "code": "254637007"}]},
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "severity": {"coding": [{"display": "Severe"}]},
        "onsetDateTime": "2023-06-01",
    }

@pytest.fixture
def sample_fhir_medication() -> dict:
    return {
        "resourceType": "MedicationRequest",
        "id": "MR001",
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [{"display": "Pembrolizumab 200mg", "code": "1860497"}]
        },
        "dosageInstruction": [{"text": "200mg IV Q3W"}],
        "authoredOn": "2023-07-01",
    }

@pytest.fixture
def sample_fhir_allergy() -> dict:
    return {
        "resourceType": "AllergyIntolerance",
        "id": "AI001",
        "code": {"coding": [{"display": "Penicillin", "code": "372687004"}]},
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "criticality": "high",
        "reaction": [
            {
                "manifestation": [{"coding": [{"display": "Anaphylaxis"}]}],
                "severity": "severe",
            }
        ],
    }

@pytest.fixture
def sample_fhir_observation() -> dict:
    return {
        "resourceType": "Observation",
        "id": "OB001",
        "code": {"coding": [{"display": "Hemoglobin", "code": "718-7"}]},
        "valueQuantity": {"value": 11.2, "unit": "g/dL"},
        "status": "final",
        "effectiveDateTime": "2024-01-15",
        "interpretation": [{"coding": [{"display": "Low"}]}],
    }

@pytest.fixture
def sample_fhir_bundle(
    sample_fhir_patient,
    sample_fhir_condition,
    sample_fhir_medication,
    sample_fhir_allergy,
    sample_fhir_observation,
) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": sample_fhir_patient},
            {"resource": sample_fhir_condition},
            {"resource": sample_fhir_medication},
            {"resource": sample_fhir_allergy},
            {"resource": sample_fhir_observation},
        ],
    }


# ── RAG Pipeline tests ────────────────────────────────────────────────────────

class TestClinicalRAGPipeline:

    @patch("src.rag.clinical_rag.Chroma")
    @patch("src.rag.clinical_rag.OpenAIEmbeddings")
    @patch("src.rag.clinical_rag.ChatOpenAI")
    def test_pipeline_initializes(self, mock_llm, mock_embeddings, mock_chroma):
        from src.rag.clinical_rag import ClinicalRAGPipeline
        pipeline = ClinicalRAGPipeline(api_key="test-key")
        assert pipeline is not None
        assert pipeline.model_name == "gpt-4o"

    @patch("src.rag.clinical_rag.Chroma")
    @patch("src.rag.clinical_rag.OpenAIEmbeddings")
    @patch("src.rag.clinical_rag.ChatOpenAI")
    def test_ingest_text(self, mock_llm, mock_embeddings, mock_chroma):
        from src.rag.clinical_rag import ClinicalRAGPipeline
        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        pipeline = ClinicalRAGPipeline(api_key="test-key")
        pipeline._vectorstore = mock_vs

        n = pipeline.ingest_text(
            "Patient John Doe has NSCLC and is on pembrolizumab.",
            metadata={"source": "discharge_note", "patient_id": "P001"},
        )
        assert isinstance(n, int)

    @patch("src.rag.clinical_rag.Chroma")
    @patch("src.rag.clinical_rag.OpenAIEmbeddings")
    @patch("src.rag.clinical_rag.ChatOpenAI")
    def test_fhir_to_text_patient(self, mock_llm, mock_embeddings, mock_chroma, sample_fhir_patient):
        from src.rag.clinical_rag import ClinicalRAGPipeline
        pipeline = ClinicalRAGPipeline(api_key="test-key")
        text = pipeline._fhir_to_text(sample_fhir_patient)
        assert "Patient" in text
        assert "John" in text
        assert "Doe" in text

    @patch("src.rag.clinical_rag.Chroma")
    @patch("src.rag.clinical_rag.OpenAIEmbeddings")
    @patch("src.rag.clinical_rag.ChatOpenAI")
    def test_fhir_to_text_allergy(self, mock_llm, mock_embeddings, mock_chroma, sample_fhir_allergy):
        from src.rag.clinical_rag import ClinicalRAGPipeline
        pipeline = ClinicalRAGPipeline(api_key="test-key")
        text = pipeline._fhir_to_text(sample_fhir_allergy)
        assert "ALLERGY" in text
        assert "Penicillin" in text

    @patch("src.rag.clinical_rag.Chroma")
    @patch("src.rag.clinical_rag.OpenAIEmbeddings")
    @patch("src.rag.clinical_rag.ChatOpenAI")
    def test_fhir_to_text_observation(self, mock_llm, mock_embeddings, mock_chroma, sample_fhir_observation):
        from src.rag.clinical_rag import ClinicalRAGPipeline
        pipeline = ClinicalRAGPipeline(api_key="test-key")
        text = pipeline._fhir_to_text(sample_fhir_observation)
        assert "Hemoglobin" in text
        assert "11.2" in text

    def test_missing_api_key_raises(self):
        from src.rag.clinical_rag import ClinicalRAGPipeline
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError):
                ClinicalRAGPipeline(api_key="")


# ── FHIR Indexer tests ────────────────────────────────────────────────────────

class TestFHIRIndexer:

    @patch("src.rag.clinical_rag.Chroma")
    @patch("src.rag.clinical_rag.OpenAIEmbeddings")
    @patch("src.rag.clinical_rag.ChatOpenAI")
    def test_index_bundle_file(
        self, mock_llm, mock_embeddings, mock_chroma,
        sample_fhir_bundle, tmp_path
    ):
        from src.rag.clinical_rag import ClinicalRAGPipeline
        from src.rag.fhir_indexer import FHIRIndexer

        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs

        bundle_file = tmp_path / "patient_P001.json"
        bundle_file.write_text(json.dumps(sample_fhir_bundle))

        pipeline = ClinicalRAGPipeline(api_key="test-key")
        pipeline._vectorstore = mock_vs

        indexer = FHIRIndexer(rag_pipeline=pipeline)
        result = indexer.index_bundle_file(bundle_file, patient_id="P001")

        assert result.patient_id == "P001"
        assert result.total_resources == 5
        assert "Patient" in result.resources_by_type
        assert "AllergyIntolerance" in result.resources_by_type

    @patch("src.rag.clinical_rag.Chroma")
    @patch("src.rag.clinical_rag.OpenAIEmbeddings")
    @patch("src.rag.clinical_rag.ChatOpenAI")
    def test_index_bundle_missing_file_raises(
        self, mock_llm, mock_embeddings, mock_chroma
    ):
        from src.rag.clinical_rag import ClinicalRAGPipeline
        from src.rag.fhir_indexer import FHIRIndexer
        pipeline = ClinicalRAGPipeline(api_key="test-key")
        indexer = FHIRIndexer(rag_pipeline=pipeline)
        with pytest.raises(FileNotFoundError):
            indexer.index_bundle_file("nonexistent.json")


# ── Embedder tests ────────────────────────────────────────────────────────────

class TestClinicalEmbedder:

    def test_clinical_section_chunking(self):
        from src.rag.embeddings import ClinicalEmbedder
        with patch("src.rag.embeddings.openai.OpenAI"):
            embedder = ClinicalEmbedder(api_key="test-key", cache_dir=None)

        text = """SUBJECTIVE: Patient reports shortness of breath.
OBJECTIVE: HR 98, BP 142/88, O2 sat 94%.
ASSESSMENT: Likely heart failure exacerbation.
PLAN: Increase furosemide, daily weights, follow up in 1 week."""

        chunks = embedder.chunk_clinical_document(text, chunk_size=200)
        assert len(chunks) >= 2
        sections = [c["section"] for c in chunks]
        assert "SUBJECTIVE" in sections or any("SUBJ" in s for s in sections)

    def test_cache_key_deterministic(self):
        from src.rag.embeddings import ClinicalEmbedder
        with patch("src.rag.embeddings.openai.OpenAI"):
            embedder = ClinicalEmbedder(api_key="test-key", cache_dir=None)
        text = "Patient has penicillin allergy."
        assert embedder._cache_key(text) == embedder._cache_key(text)

    @patch("src.rag.embeddings.openai.OpenAI")
    def test_embed_uses_cache(self, mock_openai_class):
        from src.rag.embeddings import ClinicalEmbedder
        embedder = ClinicalEmbedder(api_key="test-key", cache_dir=None)

        fake_embedding = [0.1] * 1536
        embedder._cache[embedder._cache_key("test text")] = fake_embedding

        result = embedder.embed("test text")
        assert result.cached is True
        assert result.embedding == fake_embedding


# ── Drug Interaction Checker tests ───────────────────────────────────────────

class TestDrugInteractionChecker:

    def test_warfarin_aspirin_major(self):
        from src.agents.drug_interaction_tool import DrugInteractionChecker
        checker = DrugInteractionChecker()
        result = checker.check_single_interaction("warfarin", "aspirin")
        assert result is not None
        assert result.severity == "MAJOR"

    def test_ssri_maoi_contraindicated(self):
        from src.agents.drug_interaction_tool import DrugInteractionChecker
        checker = DrugInteractionChecker()
        result = checker.check_single_interaction("ssri", "maoi")
        assert result is not None
        assert result.severity == "CONTRAINDICATED"

    def test_no_interaction_returns_none(self):
        from src.agents.drug_interaction_tool import DrugInteractionChecker
        checker = DrugInteractionChecker()
        result = checker.check_single_interaction("acetaminophen", "vitamin_c")
        assert result is None

    def test_screen_medication_list(self):
        from src.agents.drug_interaction_tool import DrugInteractionChecker
        checker = DrugInteractionChecker()
        report = checker.screen_medication_list(
            patient_id="P001",
            medications=["warfarin", "aspirin", "metformin", "lisinopril"],
            allergies=["penicillin"],
            egfr=35,
        )
        assert report.patient_id == "P001"
        assert len(report.interactions) > 0
        assert report.has_critical_alerts

    def test_allergy_alert_detected(self):
        from src.agents.drug_interaction_tool import DrugInteractionChecker
        checker = DrugInteractionChecker()
        report = checker.screen_medication_list(
            patient_id="P002",
            medications=["amoxicillin", "metformin"],
            allergies=["penicillin"],
        )
        assert len(report.allergy_alerts) > 0
        assert any("amoxicillin" in a.lower() for a in report.allergy_alerts)

    def test_renal_adjustment_metformin(self):
        from src.agents.drug_interaction_tool import DrugInteractionChecker
        checker = DrugInteractionChecker()
        report = checker.screen_medication_list(
            patient_id="P003",
            medications=["metformin"],
            egfr=25,
        )
        assert len(report.renal_adjustments) > 0

    def test_polypharmacy_score_high(self):
        from src.agents.drug_interaction_tool import DrugInteractionChecker
        checker = DrugInteractionChecker()
        report = checker.screen_medication_list(
            patient_id="P004",
            medications=["med" + str(i) for i in range(12)],
        )
        assert report.polypharmacy_score == 3

    def test_high_alert_flag(self):
        from src.agents.drug_interaction_tool import DrugInteractionChecker
        checker = DrugInteractionChecker()
        report = checker.screen_medication_list(
            patient_id="P005",
            medications=["warfarin", "metformin"],
        )
        assert len(report.high_alert_flags) > 0


# ── Treatment Recommender tests ───────────────────────────────────────────────

class TestTreatmentRecommender:

    def test_nsclc_pdl1_high(self):
        from src.recommender.treatment_recommender import (
            TreatmentRecommender, PatientProfile
        )
        recommender = TreatmentRecommender()
        profile = PatientProfile(
            patient_id="P001",
            indication="nsclc",
            biomarkers={"pdl1_tps": 75, "egfr_mut": False, "alk_fusion": False},
            egfr=80,
            performance_status=1,
        )
        report = recommender.recommend(profile)
        assert len(report.recommendations) > 0
        top = report.recommendations[0]
        assert "pembrolizumab" in top.treatment.lower() or top.score > 0.5

    def test_nsclc_egfr_positive(self):
        from src.recommender.treatment_recommender import (
            TreatmentRecommender, PatientProfile
        )
        recommender = TreatmentRecommender()
        profile = PatientProfile(
            patient_id="P002",
            indication="nsclc",
            biomarkers={"pdl1_tps": 10, "egfr_mut": True, "alk_fusion": False},
            egfr=70,
            performance_status=0,
        )
        report = recommender.recommend(profile)
        top = report.recommendations[0]
        assert "osimertinib" in top.treatment.lower()

    def test_unknown_indication(self):
        from src.recommender.treatment_recommender import (
            TreatmentRecommender, PatientProfile
        )
        recommender = TreatmentRecommender()
        profile = PatientProfile(
            patient_id="P003",
            indication="unknown_disease",
            biomarkers={},
        )
        report = recommender.recommend(profile)
        assert len(report.recommendations) == 0

    def test_contraindication_penalizes_score(self):
        from src.recommender.treatment_recommender import (
            TreatmentRecommender, PatientProfile
        )
        recommender = TreatmentRecommender()
        profile = PatientProfile(
            patient_id="P004",
            indication="type2_diabetes",
            biomarkers={"egfr": 25},
            egfr=25,
        )
        report = recommender.recommend(profile)
        metformin = next(
            (r for r in report.recommendations
             if "metformin" in r.treatment.lower()), None
        )
        if metformin:
            assert len(metformin.contraindications) > 0

    def test_collaborative_filtering_empty_db(self):
        from src.recommender.treatment_recommender import (
            TreatmentRecommender, PatientProfile
        )
        recommender = TreatmentRecommender()
        profile = PatientProfile(
            patient_id="P005",
            indication="nsclc",
            biomarkers={"pdl1_tps": 50},
        )
        similar = recommender.get_similar_patient_outcomes(profile)
        assert similar == []


# ── Ranking Engine tests ──────────────────────────────────────────────────────

class TestRankingEngines:

    @pytest.fixture
    def sample_criteria(self):
        from src.recommender.ranking_engine import TreatmentCriteria
        return [
            TreatmentCriteria("Drug A", efficacy_score=0.9, safety_score=0.7,
                               biomarker_fit_score=1.0, qol_score=0.8, guideline_score=1.0),
            TreatmentCriteria("Drug B", efficacy_score=0.6, safety_score=0.9,
                               biomarker_fit_score=0.5, qol_score=0.9, guideline_score=0.8),
            TreatmentCriteria("Drug C", efficacy_score=0.3, safety_score=0.95,
                               biomarker_fit_score=0.2, qol_score=0.95, guideline_score=0.6),
        ]

    def test_weighted_scoring_ranks(self, sample_criteria):
        from src.recommender.ranking_engine import WeightedScoringRanker
        ranker = WeightedScoringRanker()
        result = ranker.rank("P001", "nsclc", sample_criteria)
        assert len(result.ranked_treatments) == 3
        assert result.ranked_treatments[0].rank == 1
        assert result.ranked_treatments[0].treatment_name == "Drug A"

    def test_topsis_ranks(self, sample_criteria):
        from src.recommender.ranking_engine import TOPSISRanker
        ranker = TOPSISRanker()
        result = ranker.rank("P001", "nsclc", sample_criteria)
        assert len(result.ranked_treatments) == 3
        assert result.method == "topsis"
        scores = [r.composite_score for r in result.ranked_treatments]
        assert scores == sorted(scores, reverse=True)

    def test_ensemble_ranks(self, sample_criteria):
        from src.recommender.ranking_engine import EnsembleRanker
        ranker = EnsembleRanker()
        result = ranker.rank("P001", "nsclc", sample_criteria)
        assert len(result.ranked_treatments) == 3
        assert result.method == "ensemble"
        assert result.ranked_treatments[0].rank == 1

    def test_patient_centered_weights(self, sample_criteria):
        from src.recommender.ranking_engine import WeightedScoringRanker
        ranker = WeightedScoringRanker(patient_centered=True)
        result = ranker.rank("P001", "nsclc", sample_criteria)
        assert result.weights_used["safety"] > result.weights_used["efficacy"]

    def test_criteria_to_vector(self):
        from src.recommender.ranking_engine import TreatmentCriteria
        tc = TreatmentCriteria("Test", efficacy_score=0.8, safety_score=0.7,
                                biomarker_fit_score=0.9, qol_score=0.6, guideline_score=1.0)
        vec = tc.to_vector()
        assert len(vec) == 5
        assert vec[0] == 0.8

    def test_top_n_returns_correct_count(self, sample_criteria):
        from src.recommender.ranking_engine import EnsembleRanker
        ranker = EnsembleRanker()
        result = ranker.rank("P001", "nsclc", sample_criteria)
        top2 = result.top_n(2)
        assert len(top2) == 2


# ── FastAPI endpoint tests ────────────────────────────────────────────────────

class TestFastAPIEndpoints:

    @pytest.fixture
    def client(self):
        from src.serving.api import app
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "healthcare-ai-platform"

    def test_readiness_check(self, client):
        response = client.get("/ready")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "checks" in data

    def test_unauthorized_request(self, client):
        response = client.post(
            "/api/v1/drug-safety/screen",
            json={"patient_id": "P001", "medications": ["warfarin"]},
        )
        assert response.status_code == 401

    def test_drug_safety_endpoint(self, client):
        response = client.post(
            "/api/v1/drug-safety/screen",
            headers={"Authorization": "Bearer dev-key-123"},
            json={
                "patient_id": "P001",
                "medications": ["warfarin", "aspirin", "metformin"],
                "allergies": ["penicillin"],
                "egfr": 45,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "P001"
        assert data["n_medications"] == 3
        assert data["n_interactions"] >= 0
        assert "request_id" in data

    def test_recommendation_endpoint(self, client):
        response = client.post(
            "/api/v1/recommend",
            headers={"Authorization": "Bearer dev-key-123"},
            json={
                "patient_id": "P001",
                "indication": "nsclc",
                "biomarkers": {"pdl1_tps": 60, "egfr_mut": False},
                "egfr": 75,
                "performance_status": 1,
                "treatment_line": "first",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "P001"
        assert data["indication"] == "nsclc"
        assert len(data["top_recommendations"]) > 0

    def test_request_id_returned(self, client):
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_invalid_indication_returns_empty(self, client):
        response = client.post(
            "/api/v1/recommend",
            headers={"Authorization": "Bearer dev-key-123"},
            json={
                "patient_id": "P999",
                "indication": "unknown_condition",
                "biomarkers": {},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["n_options"] == 0

    @patch("src.agents.summarizer_agent.ChatOpenAI")
    def test_summarize_endpoint(self, mock_chat, client):
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "SITUATION: Patient review.\nBACKGROUND: History.\nASSESSMENT: Stable.\nRECOMMENDATION: Continue."
        mock_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_instance

        response = client.post(
            "/api/v1/summarize",
            headers={"Authorization": "Bearer dev-key-123"},
            json={
                "patient_id": "P001",
                "clinical_text": "Patient has non-small cell lung cancer on pembrolizumab. Stable clinical status without acute complications.",
                "summary_type": "sbar",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "P001"
        assert data["summary_type"] == "sbar"
        assert len(data["summary_text"]) > 0

    @patch("src.agents.clinical_agent.ChatOpenAI")
    def test_agent_run_endpoint(self, mock_chat, client):
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "No critical contraindications noted."
        mock_instance.invoke.return_value = mock_response
        mock_instance.bind_tools.return_value = mock_instance
        mock_chat.return_value = mock_instance

        response = client.post(
            "/api/v1/agents/run",
            headers={"Authorization": "Bearer dev-key-123"},
            json={
                "patient_id": "P001",
                "query": "Check drug safety for warfarin and aspirin",
                "patient_context": "Patient on warfarin 5mg daily.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "P001"
        assert data["query_type"] == "drug_safety"

    @patch("src.rag.clinical_rag.Chroma")
    @patch("src.rag.clinical_rag.OpenAIEmbeddings")
    @patch("src.rag.clinical_rag.ClinicalRAGPipeline.query")
    def test_rag_ingest_and_query_endpoints(self, mock_query, mock_embeddings, mock_chroma, client):
        mock_vs = MagicMock()
        mock_chroma.return_value = mock_vs
        mock_vs.similarity_search.return_value = []

        from src.rag.clinical_rag import RAGResponse
        mock_query.return_value = RAGResponse(
            question="What is the diagnosis?",
            answer="Patient is diagnosed with NSCLC.",
            source_documents=["doc1"],
            n_chunks_retrieved=1,
            model_used="gpt-4o",
            confidence_note="Test confidence",
        )

        # Test Ingest
        ingest_resp = client.post(
            "/api/v1/rag/ingest",
            headers={"Authorization": "Bearer dev-key-123"},
            json={
                "patient_id": "P001",
                "text": "Patient John Doe is undergoing oncology evaluation. Laboratory values are stable.",
                "source": "clinical_note",
            },
        )
        assert ingest_resp.status_code == 200
        assert ingest_resp.json()["status"] == "ingested"

        # Test Query
        query_resp = client.post(
            "/api/v1/rag/query",
            headers={"Authorization": "Bearer dev-key-123"},
            json={
                "patient_id": "P001",
                "question": "What is the diagnosis?",
            },
        )
        assert query_resp.status_code == 200
        assert query_resp.json()["answer"] == "Patient is diagnosed with NSCLC."


# ── Additional Module Unit Tests ──────────────────────────────────────────────

class TestSummarizerAgent:

    @patch("src.agents.summarizer_agent.ChatOpenAI")
    def test_summarize_sbar(self, mock_chat):
        from src.agents.summarizer_agent import ClinicalSummarizerAgent
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(
            content="SITUATION: Patient Doe.\nBACKGROUND: NSCLC.\nASSESSMENT: Improving.\nRECOMMENDATION: Discharge."
        )
        mock_chat.return_value = mock_instance

        agent = ClinicalSummarizerAgent(api_key="test-key")
        summary = agent.summarize("Patient is recovering well after first cycle.", patient_id="P001", summary_type="sbar")
        assert summary.patient_id == "P001"
        assert summary.summary_type == "sbar"
        assert summary.word_count > 0


class TestTherapistMatcher:

    def test_therapist_matcher_initialization_and_scoring(self):
        from src.recommender.therapist_matcher import (
            TherapistMatcher, PatientMentalHealthProfile, TherapistProfile
        )
        matcher = TherapistMatcher()
        patient = PatientMentalHealthProfile(
            patient_id="P_MH_01",
            phq9_score=14,
            gad7_score=11,
            primary_concerns=["depression", "anxiety"],
            preferred_language="english",
            telehealth_ok=True,
        )
        therapists = [
            TherapistProfile(
                therapist_id="T01",
                name="Dr. Smith",
                credentials="PhD",
                specialisations=["depression", "anxiety"],
                treatment_approaches=["cbt"],
                languages=["english"],
                accepts_new_patients=True,
                available_slots_per_week=5,
            ),
            TherapistProfile(
                therapist_id="T02",
                name="Dr. Taylor",
                credentials="LCSW",
                specialisations=["trauma"],
                treatment_approaches=["emdr"],
                languages=["english"],
                accepts_new_patients=True,
                available_slots_per_week=2,
            ),
        ]
        report = matcher.match(patient, therapists)
        assert len(report.matches) > 0
        top = report.top_matches(1)
        assert len(top) == 1
        assert top[0].therapist_id == "T01"


class TestTrialMatcher:

    def test_trial_matcher_matching(self):
        from src.trial_matching.trial_matcher import (
            ClinicalTrialMatcher, PatientProfile, ClinicalTrial, EligibilityCriterion
        )
        matcher = ClinicalTrialMatcher()
        patient = PatientProfile(
            patient_id="P_TR_01",
            age=55,
            diagnoses=["C34.90"],  # NSCLC
            biomarkers=["PD-L1", "EGFR-wt"],
            ecog_status=1,
            lab_values={"egfr": 75},
        )
        trial = ClinicalTrial(
            trial_id="NCT00000001",
            title="Phase 3 Immunotherapy in Advanced NSCLC",
            phase="Phase 3",
            indication="NSCLC",
            sponsor="Test Pharma",
            status="Recruiting",
            inclusion_criteria=[
                EligibilityCriterion(
                    criterion_id="C01",
                    criterion_type="age",
                    description="Age >= 18",
                    is_inclusion=True,
                    value_min=18,
                ),
                EligibilityCriterion(
                    criterion_id="C02",
                    criterion_type="biomarker",
                    description="PD-L1 positive",
                    is_inclusion=True,
                    required_values=["PD-L1"],
                ),
            ],
        )
        report = matcher.match_patient(patient, [trial])
        assert len(report.matches) > 0
        assert report.matches[0].match_status == "ELIGIBLE"


class TestAIGovernance:

    def test_pre_deployment_check_and_audit(self):
        from src.governance.ai_governance import AIGovernanceFramework, RESPONSIBLE_AI_CHECKLIST
        framework = AIGovernanceFramework(organisation="Test Health")
        checklist = {item: True for item in RESPONSIBLE_AI_CHECKLIST}
        perf = {"recall": 0.85, "auc": 0.88, "precision": 0.75}
        assessment = framework.run_pre_deployment_check(
            model_name="test_model",
            model_version="1.0.0",
            use_case="clinical_decision_support",
            checklist=checklist,
            performance_metrics=perf,
        )
        assert assessment.approved_for_deployment is True
        assert assessment.risk_level in ["low", "moderate", "high", "critical"]

