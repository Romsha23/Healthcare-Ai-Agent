"""
rag/fhir_indexer.py
────────────────────
FHIR R4 resource indexer for the clinical RAG pipeline.

Handles bulk ingestion of FHIR R4 resources from:
    - FHIR REST API endpoints (Epic, Cerner, Oracle Health EHR)
    - FHIR Bundle JSON files
    - Synthetic patient datasets (Synthea)

Supports all key clinical resource types:
    Patient, Condition, MedicationRequest, Observation,
    AllergyIntolerance, Procedure, DiagnosticReport,
    Encounter, CarePlan, Immunization

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from .clinical_rag import ClinicalRAGPipeline

logger = logging.getLogger(__name__)


# ── FHIR resource priority ────────────────────────────────────────────────────

# Resource types in clinical priority order
PRIORITY_RESOURCES = [
    "AllergyIntolerance",   # Safety-critical — always index first
    "MedicationRequest",    # Active medications
    "Condition",            # Active problem list
    "Observation",          # Lab results, vitals
    "Procedure",            # Procedures performed
    "DiagnosticReport",     # Lab/imaging reports
    "Encounter",            # Clinical encounters
    "CarePlan",             # Care plans
    "Immunization",         # Vaccination history
    "Patient",              # Demographics (last — least dynamic)
]

# FHIR search parameters per resource type
FHIR_SEARCH_PARAMS = {
    "Condition":          "?patient={patient_id}&clinical-status=active",
    "MedicationRequest":  "?patient={patient_id}&status=active",
    "Observation":        "?patient={patient_id}&_sort=-date&_count=50",
    "AllergyIntolerance": "?patient={patient_id}",
    "Procedure":          "?patient={patient_id}&_sort=-date&_count=30",
    "DiagnosticReport":   "?patient={patient_id}&_sort=-date&_count=20",
    "Encounter":          "?patient={patient_id}&_sort=-date&_count=10",
    "CarePlan":           "?patient={patient_id}&status=active",
    "Immunization":       "?patient={patient_id}",
}


# ── Indexing result ───────────────────────────────────────────────────────────

@dataclass
class IndexingResult:
    """Result of a FHIR indexing operation."""
    patient_id: str
    total_resources: int = 0
    total_chunks: int = 0
    resources_by_type: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    success: bool = True

    def summary(self) -> dict:
        return {
            "patient_id":        self.patient_id,
            "total_resources":   self.total_resources,
            "total_chunks":      self.total_chunks,
            "resources_by_type": self.resources_by_type,
            "errors":            self.errors,
            "success":           self.success,
        }


# ── FHIR Indexer ─────────────────────────────────────────────────────────────

class FHIRIndexer:
    """
    FHIR R4 resource indexer for the clinical RAG pipeline.

    Connects to FHIR REST API endpoints or reads local FHIR
    Bundle files and indexes all clinical resources into the
    ChromaDB vector store for RAG retrieval.

    Compatible with:
        - Oracle Health (Cerner) FHIR R4 API
        - Epic MyChart FHIR R4 API
        - Google Cloud Healthcare FHIR API
        - AWS HealthLake FHIR API
        - Synthea synthetic patient data

    Parameters
    ----------
    rag_pipeline : ClinicalRAGPipeline
        Initialized RAG pipeline to index into.
    fhir_base_url : str, optional
        Base URL of the FHIR R4 server (e.g. 'https://fhir.example.com/r4').
    access_token : str, optional
        OAuth 2.0 bearer token for FHIR API authentication.

    Examples
    --------
    >>> pipeline = ClinicalRAGPipeline()
    >>> indexer = FHIRIndexer(
    ...     rag_pipeline=pipeline,
    ...     fhir_base_url="https://fhir.example.com/r4",
    ...     access_token="eyJ...",
    ... )
    >>> result = indexer.index_patient("Patient/P001")
    >>> print(result.summary())
    """

    def __init__(
        self,
        rag_pipeline: ClinicalRAGPipeline,
        fhir_base_url: Optional[str] = None,
        access_token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.pipeline      = rag_pipeline
        self.fhir_base_url = fhir_base_url.rstrip("/") if fhir_base_url else None
        self.access_token  = access_token
        self.timeout       = timeout

        self._headers = {"Accept": "application/fhir+json"}
        if access_token:
            self._headers["Authorization"] = f"Bearer {access_token}"

        logger.info(
            "FHIRIndexer initialized: server=%s",
            fhir_base_url or "local only",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def index_patient(
        self,
        patient_id: str,
        resource_types: Optional[list[str]] = None,
    ) -> IndexingResult:
        """
        Index all clinical records for a patient from the FHIR API.

        Parameters
        ----------
        patient_id : str
            FHIR patient ID (e.g. 'P001' or 'Patient/P001').
        resource_types : list[str], optional
            Resource types to index. Defaults to all PRIORITY_RESOURCES.

        Returns
        -------
        IndexingResult
        """
        if not self.fhir_base_url:
            raise RuntimeError(
                "FHIR base URL not configured. Pass fhir_base_url= to FHIRIndexer."
            )

        pid = patient_id.replace("Patient/", "")
        resource_types = resource_types or PRIORITY_RESOURCES

        result = IndexingResult(patient_id=pid)
        logger.info("Indexing patient %s: %d resource types", pid, len(resource_types))

        # Index Patient resource first for demographics
        try:
            patient = self._fetch_resource("Patient", pid)
            if patient:
                n = self.pipeline.ingest_fhir_resource(patient, patient_id=pid)
                result.resources_by_type["Patient"] = 1
                result.total_resources += 1
                result.total_chunks += n
        except Exception as e:
            result.errors.append(f"Patient demographics: {e}")

        # Index clinical resources in priority order
        for resource_type in resource_types:
            if resource_type == "Patient":
                continue
            try:
                resources = self._search_resources(resource_type, pid)
                for resource in resources:
                    n = self.pipeline.ingest_fhir_resource(resource, patient_id=pid)
                    result.total_chunks += n

                if resources:
                    result.resources_by_type[resource_type] = len(resources)
                    result.total_resources += len(resources)
                    logger.info(
                        "Indexed %d %s resources for patient %s",
                        len(resources), resource_type, pid,
                    )
            except Exception as e:
                error_msg = f"{resource_type}: {e}"
                result.errors.append(error_msg)
                logger.warning("Error indexing %s for patient %s: %s", resource_type, pid, e)

        if result.errors:
            result.success = len(result.errors) < len(resource_types) / 2

        logger.info(
            "Patient %s indexed: %d resources, %d chunks, %d errors",
            pid, result.total_resources, result.total_chunks, len(result.errors),
        )
        return result

    def index_bundle_file(
        self,
        bundle_path: str | Path,
        patient_id: Optional[str] = None,
    ) -> IndexingResult:
        """
        Index all resources from a FHIR Bundle JSON file.

        Parameters
        ----------
        bundle_path : str or Path
            Path to a FHIR R4 Bundle JSON file.
        patient_id : str, optional
            Patient identifier to attach to all indexed chunks.

        Returns
        -------
        IndexingResult
        """
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle file not found: {bundle_path}")

        with open(bundle_path, encoding="utf-8") as f:
            bundle = json.load(f)

        pid = patient_id or bundle_path.stem
        result = IndexingResult(patient_id=pid)

        entries = bundle.get("entry", [])
        logger.info(
            "Indexing bundle file '%s': %d entries", bundle_path.name, len(entries)
        )

        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType", "Unknown")
            try:
                n = self.pipeline.ingest_fhir_resource(resource, patient_id=pid)
                result.total_chunks += n
                result.resources_by_type[resource_type] = (
                    result.resources_by_type.get(resource_type, 0) + 1
                )
                result.total_resources += 1
            except Exception as e:
                result.errors.append(f"{resource_type}/{resource.get('id', '?')}: {e}")

        result.success = len(result.errors) < len(entries) / 2
        logger.info(
            "Bundle indexed: %d resources → %d chunks",
            result.total_resources, result.total_chunks,
        )
        return result

    def index_synthea_directory(
        self,
        directory: str | Path,
        max_patients: Optional[int] = None,
    ) -> list[IndexingResult]:
        """
        Index all Synthea-generated patient bundle files in a directory.

        Synthea produces realistic synthetic FHIR R4 patient data
        widely used for healthcare AI development and testing.

        Parameters
        ----------
        directory : str or Path
            Directory containing Synthea JSON bundle files.
        max_patients : int, optional
            Maximum number of patients to index. Default indexes all.

        Returns
        -------
        list[IndexingResult]
            One IndexingResult per patient bundle file.
        """
        directory = Path(directory)
        bundle_files = sorted(directory.glob("*.json"))

        if max_patients:
            bundle_files = bundle_files[:max_patients]

        logger.info(
            "Indexing Synthea directory: %d patient files", len(bundle_files)
        )

        results = []
        for i, bundle_file in enumerate(bundle_files, 1):
            logger.info(
                "Indexing patient %d/%d: %s", i, len(bundle_files), bundle_file.name
            )
            try:
                result = self.index_bundle_file(bundle_file)
                results.append(result)
            except Exception as e:
                logger.error("Failed to index %s: %s", bundle_file.name, e)
                results.append(IndexingResult(
                    patient_id=bundle_file.stem,
                    success=False,
                    errors=[str(e)],
                ))

        successful = sum(1 for r in results if r.success)
        total_chunks = sum(r.total_chunks for r in results)
        logger.info(
            "Synthea indexing complete: %d/%d patients, %d total chunks",
            successful, len(results), total_chunks,
        )
        return results

    def index_guideline_document(
        self,
        file_path: str | Path,
        guideline_name: str,
        organization: str = "",
        version: str = "",
    ) -> int:
        """
        Index a clinical practice guideline document.

        Parameters
        ----------
        file_path : str or Path
            Path to guideline PDF or text file.
        guideline_name : str
            Name of the guideline (e.g. 'NCCN NSCLC Guidelines v2024').
        organization : str
            Issuing organization (e.g. 'NCCN', 'ASCO', 'AHA').
        version : str
            Guideline version or year.

        Returns
        -------
        int
            Number of chunks indexed.
        """
        metadata = {
            "source":       guideline_name,
            "doc_type":     "clinical_guideline",
            "organization": organization,
            "version":      version,
        }
        n = self.pipeline.ingest_file(file_path, metadata)
        logger.info(
            "Guideline indexed: '%s' → %d chunks", guideline_name, n
        )
        return n

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_resource(self, resource_type: str, resource_id: str) -> Optional[dict]:
        """Fetch a single FHIR resource by type and ID."""
        url = f"{self.fhir_base_url}/{resource_type}/{resource_id}"
        response = requests.get(url, headers=self._headers, timeout=self.timeout)
        if response.status_code == 200:
            return response.json()
        logger.warning("FHIR fetch failed: %s %s", response.status_code, url)
        return None

    def _search_resources(
        self, resource_type: str, patient_id: str
    ) -> list[dict]:
        """Search for FHIR resources for a patient."""
        params = FHIR_SEARCH_PARAMS.get(resource_type, f"?patient={patient_id}")
        url = f"{self.fhir_base_url}/{resource_type}{params.format(patient_id=patient_id)}"

        resources = []
        while url:
            response = requests.get(url, headers=self._headers, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning("FHIR search failed: %s %s", response.status_code, url)
                break

            bundle = response.json()
            entries = bundle.get("entry", [])
            resources.extend(
                entry["resource"] for entry in entries if "resource" in entry
            )

            # Follow pagination
            next_url = next(
                (link["url"] for link in bundle.get("link", [])
                 if link.get("relation") == "next"),
                None,
            )
            url = next_url

        return resources
