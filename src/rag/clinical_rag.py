"""
rag/clinical_rag.py
────────────────────
Production-grade RAG pipeline for clinical document retrieval
and question answering.

Supports retrieval-augmented generation over:
    - FHIR R4 patient records (Patient, Condition, MedicationRequest,
      Observation, AllergyIntolerance, Procedure)
    - Clinical practice guidelines (PDF/text)
    - Drug prescribing information
    - Discharge summaries and clinical notes

Pipeline:
    Documents → Chunk → Embed → Store (ChromaDB)
    Query → Embed → Retrieve → Rerank → Generate (GPT-4o)

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


# ── Clinical prompt templates ─────────────────────────────────────────────────

CLINICAL_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert clinical decision support AI assistant operating within
a healthcare system. You answer clinical questions using only the provided
context from patient records and clinical guidelines.

CRITICAL RULES:
- Base answers ONLY on the provided context — never fabricate clinical data
- Always cite the source document for each clinical fact you state
- If the context does not contain sufficient information, explicitly say so
- Flag any urgent clinical concerns (allergies, drug interactions, critical values)
- This is a decision SUPPORT tool — final clinical decisions rest with clinicians
- Never provide definitive diagnoses — use language like "the records indicate" or "per the guidelines"

Context from patient records and clinical guidelines:
{context}

Clinical question: {question}

Provide a structured clinical response:
""")

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are a clinical documentation specialist. Summarize the following
patient information concisely for a clinical handoff.

Patient records:
{context}

Produce a structured SBAR summary (Situation, Background, Assessment, Recommendation).
Keep it under 300 words. Flag any urgent items clearly.
""")


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Response from the clinical RAG pipeline."""
    question: str
    answer: str
    source_documents: list[str] = field(default_factory=list)
    n_chunks_retrieved: int = 0
    model_used: str = ""
    confidence_note: str = ""

    def to_dict(self) -> dict:
        return {
            "question":           self.question,
            "answer":             self.answer,
            "sources":            self.source_documents,
            "chunks_retrieved":   self.n_chunks_retrieved,
            "model":              self.model_used,
            "confidence_note":    self.confidence_note,
        }


@dataclass
class IndexStats:
    """Statistics about the current vector store index."""
    total_documents: int = 0
    total_chunks: int = 0
    document_types: dict[str, int] = field(default_factory=dict)
    collection_name: str = ""


# ── Clinical RAG Pipeline ─────────────────────────────────────────────────────

class ClinicalRAGPipeline:
    """
    Production-grade RAG pipeline for clinical document retrieval
    and AI-powered question answering.

    Ingests FHIR R4 records, clinical guidelines, and unstructured
    clinical notes into a ChromaDB vector store. Answers clinical
    queries using GPT-4o with retrieved context, source citations,
    and clinical safety guardrails.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env variable.
    persist_dir : str
        Directory for ChromaDB persistence. Default './chroma_db'.
    collection_name : str
        ChromaDB collection name. Default 'clinical_docs'.
    model : str
        LLM model for generation. Default 'gpt-4o'.
    embedding_model : str
        OpenAI embedding model. Default 'text-embedding-3-small'.
    chunk_size : int
        Document chunk size in characters. Default 800.
    chunk_overlap : int
        Overlap between chunks. Default 100.
    n_retrieve : int
        Number of chunks to retrieve per query. Default 6.

    Examples
    --------
    >>> pipeline = ClinicalRAGPipeline()
    >>> pipeline.ingest_text(
    ...     text="Patient John Doe, 65M, with NSCLC...",
    ...     metadata={"source": "discharge_summary", "patient_id": "P001"},
    ... )
    >>> response = pipeline.query("What medications is the patient on?")
    >>> print(response.answer)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        persist_dir: str = "./chroma_db",
        collection_name: str = "clinical_docs",
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        n_retrieve: int = 6,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key=."
            )

        self.model_name     = model
        self.n_retrieve     = n_retrieve
        self.collection_name = collection_name

        # Embeddings
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key,
        )

        # Vector store
        self._vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )

        # Text splitter — clinical-optimized chunking
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # LLM
        self._llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=api_key,
        )

        # Build RAG chain
        self._rag_chain = self._build_rag_chain()

        logger.info(
            "ClinicalRAGPipeline initialized: model=%s, collection=%s",
            model, collection_name,
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Ingest a text document into the vector store.

        Parameters
        ----------
        text : str
            Document text content.
        metadata : dict, optional
            Document metadata (source, patient_id, doc_type, date, etc.)

        Returns
        -------
        int
            Number of chunks added to the vector store.
        """
        metadata = metadata or {}
        doc = Document(page_content=text, metadata=metadata)
        chunks = self._splitter.split_documents([doc])

        self._vectorstore.add_documents(chunks)
        logger.info(
            "Ingested '%s': %d chunks",
            metadata.get("source", "unknown"), len(chunks),
        )
        return len(chunks)

    def ingest_file(
        self,
        file_path: str | Path,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Ingest a text or PDF file into the vector store.

        Parameters
        ----------
        file_path : str or Path
            Path to the document file (.txt or .pdf).
        metadata : dict, optional
            Additional metadata to attach to all chunks.

        Returns
        -------
        int
            Number of chunks added.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = metadata or {}
        metadata["source"] = file_path.name
        metadata["file_path"] = str(file_path)

        if file_path.suffix.lower() == ".pdf":
            text = self._extract_pdf_text(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")

        return self.ingest_text(text, metadata)

    def ingest_fhir_resource(
        self,
        fhir_resource: dict,
        patient_id: Optional[str] = None,
    ) -> int:
        """
        Ingest a FHIR R4 resource into the vector store.

        Converts structured FHIR JSON to searchable clinical text
        and indexes it with FHIR-specific metadata.

        Parameters
        ----------
        fhir_resource : dict
            FHIR R4 resource as a Python dictionary.
        patient_id : str, optional
            Patient identifier for metadata tagging.

        Returns
        -------
        int
            Number of chunks added.
        """
        resource_type = fhir_resource.get("resourceType", "Unknown")
        text = self._fhir_to_text(fhir_resource)

        metadata = {
            "source":        f"FHIR_{resource_type}",
            "resource_type": resource_type,
            "fhir_id":       fhir_resource.get("id", ""),
            "patient_id":    patient_id or "",
            "doc_type":      "fhir_record",
        }

        logger.info("Ingesting FHIR %s resource", resource_type)
        return self.ingest_text(text, metadata)

    def ingest_patient_bundle(
        self,
        fhir_bundle: dict,
        patient_id: Optional[str] = None,
    ) -> dict:
        """
        Ingest all resources from a FHIR R4 Bundle.

        Parameters
        ----------
        fhir_bundle : dict
            FHIR R4 Bundle resource containing multiple entries.
        patient_id : str, optional
            Patient identifier.

        Returns
        -------
        dict
            Summary of ingestion results by resource type.
        """
        entries = fhir_bundle.get("entry", [])
        summary: dict[str, int] = {}

        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType", "Unknown")
            n_chunks = self.ingest_fhir_resource(resource, patient_id)
            summary[resource_type] = summary.get(resource_type, 0) + n_chunks

        total = sum(summary.values())
        logger.info(
            "Bundle ingested: %d resources → %d chunks — %s",
            len(entries), total, summary,
        )
        return summary

    # ── Retrieval & Generation ────────────────────────────────────────────────

    def query(
        self,
        question: str,
        patient_id: Optional[str] = None,
        filter_metadata: Optional[dict] = None,
    ) -> RAGResponse:
        """
        Answer a clinical question using retrieved context.

        Parameters
        ----------
        question : str
            Clinical question (e.g. "What medications is the patient on?",
            "Are there any drug allergies documented?").
        patient_id : str, optional
            If provided, filters retrieval to this patient's records.
        filter_metadata : dict, optional
            Additional ChromaDB metadata filters.

        Returns
        -------
        RAGResponse
            Structured response with answer and source citations.
        """
        # Build filter
        where_filter = None
        if patient_id:
            where_filter = {"patient_id": patient_id}
        if filter_metadata:
            where_filter = {**(where_filter or {}), **filter_metadata}

        # Retrieve relevant chunks
        retriever = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.n_retrieve,
                "fetch_k": self.n_retrieve * 3,
                **({"filter": where_filter} if where_filter else {}),
            },
        )

        retrieved_docs = retriever.get_relevant_documents(question)
        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in retrieved_docs
        })

        # Generate answer
        context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in retrieved_docs
        )

        chain_input = {"context": context, "question": question}
        answer = self._rag_chain.invoke(chain_input)

        response = RAGResponse(
            question=question,
            answer=answer,
            source_documents=sources,
            n_chunks_retrieved=len(retrieved_docs),
            model_used=self.model_name,
            confidence_note=(
                "Based on retrieved clinical records. "
                "Verify with primary source documentation."
            ),
        )

        logger.info(
            "Query answered: %d chunks retrieved, %d sources — '%s...'",
            len(retrieved_docs), len(sources), question[:50],
        )
        return response

    def summarize_patient(
        self,
        patient_id: str,
    ) -> RAGResponse:
        """
        Generate an SBAR clinical summary for a patient.

        Parameters
        ----------
        patient_id : str
            Patient identifier to summarize.

        Returns
        -------
        RAGResponse
            SBAR-structured patient summary.
        """
        # Retrieve all chunks for this patient
        results = self._vectorstore.similarity_search(
            query=f"patient {patient_id} clinical summary",
            k=10,
            filter={"patient_id": patient_id},
        )

        if not results:
            return RAGResponse(
                question=f"SBAR summary for patient {patient_id}",
                answer=f"No clinical records found for patient {patient_id}.",
                confidence_note="No records indexed for this patient.",
            )

        context = "\n\n---\n\n".join(
            doc.page_content for doc in results
        )

        summary_chain = (
            {"context": RunnablePassthrough()}
            | SUMMARY_PROMPT
            | self._llm
            | StrOutputParser()
        )
        answer = summary_chain.invoke(context)

        return RAGResponse(
            question=f"SBAR summary for patient {patient_id}",
            answer=answer,
            source_documents=list({
                d.metadata.get("source", "unknown") for d in results
            }),
            n_chunks_retrieved=len(results),
            model_used=self.model_name,
        )

    def get_index_stats(self) -> IndexStats:
        """Return statistics about the current vector store index."""
        collection = self._vectorstore._collection
        count = collection.count()

        return IndexStats(
            total_chunks=count,
            collection_name=self.collection_name,
        )

    def clear_index(self) -> None:
        """Clear all documents from the vector store."""
        self._vectorstore._collection.delete(
            where={"source": {"$ne": "__placeholder__"}}
        )
        logger.info("Vector store cleared.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_rag_chain(self):
        """Build the LangChain RAG chain."""
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | CLINICAL_RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

    @staticmethod
    def _fhir_to_text(resource: dict) -> str:
        """Convert a FHIR R4 resource to searchable clinical text."""
        resource_type = resource.get("resourceType", "")
        lines = [f"FHIR Resource: {resource_type}"]

        if resource_type == "Patient":
            name = resource.get("name", [{}])[0]
            given = " ".join(name.get("given", []))
            family = name.get("family", "")
            lines.append(f"Patient name: {given} {family}")
            lines.append(f"Date of birth: {resource.get('birthDate', 'unknown')}")
            lines.append(f"Gender: {resource.get('gender', 'unknown')}")
            for id_entry in resource.get("identifier", []):
                lines.append(f"Identifier: {id_entry.get('value', '')} ({id_entry.get('system', '')})")

        elif resource_type == "Condition":
            code = resource.get("code", {})
            coding = code.get("coding", [{}])[0]
            lines.append(f"Condition: {coding.get('display', code.get('text', 'unknown'))}")
            lines.append(f"Clinical status: {resource.get('clinicalStatus', {}).get('coding', [{}])[0].get('code', 'unknown')}")
            lines.append(f"Onset: {resource.get('onsetDateTime', resource.get('onsetAge', {}).get('value', 'unknown'))}")
            lines.append(f"Severity: {resource.get('severity', {}).get('coding', [{}])[0].get('display', 'unknown')}")

        elif resource_type == "MedicationRequest":
            med = resource.get("medicationCodeableConcept", {})
            coding = med.get("coding", [{}])[0]
            lines.append(f"Medication: {coding.get('display', med.get('text', 'unknown'))}")
            lines.append(f"Status: {resource.get('status', 'unknown')}")
            lines.append(f"Intent: {resource.get('intent', 'unknown')}")
            dosage = resource.get("dosageInstruction", [{}])[0]
            lines.append(f"Dosage: {dosage.get('text', 'see instructions')}")
            lines.append(f"Authored: {resource.get('authoredOn', 'unknown')}")

        elif resource_type == "Observation":
            code = resource.get("code", {})
            coding = code.get("coding", [{}])[0]
            lines.append(f"Observation: {coding.get('display', code.get('text', 'unknown'))}")
            value = resource.get("valueQuantity", {})
            lines.append(f"Value: {value.get('value', 'N/A')} {value.get('unit', '')}")
            lines.append(f"Status: {resource.get('status', 'unknown')}")
            lines.append(f"Effective date: {resource.get('effectiveDateTime', 'unknown')}")
            interp = resource.get("interpretation", [{}])[0].get("coding", [{}])[0]
            lines.append(f"Interpretation: {interp.get('display', 'N/A')}")

        elif resource_type == "AllergyIntolerance":
            substance = resource.get("code", {})
            coding = substance.get("coding", [{}])[0]
            lines.append(f"ALLERGY: {coding.get('display', substance.get('text', 'unknown'))}")
            lines.append(f"Clinical status: {resource.get('clinicalStatus', {}).get('coding', [{}])[0].get('code', 'unknown')}")
            lines.append(f"Criticality: {resource.get('criticality', 'unknown')}")
            for reaction in resource.get("reaction", []):
                manifestations = [
                    m.get("coding", [{}])[0].get("display", "")
                    for m in reaction.get("manifestation", [])
                ]
                lines.append(f"Reaction: {', '.join(manifestations)}")
                lines.append(f"Severity: {reaction.get('severity', 'unknown')}")

        elif resource_type == "Procedure":
            code = resource.get("code", {})
            coding = code.get("coding", [{}])[0]
            lines.append(f"Procedure: {coding.get('display', code.get('text', 'unknown'))}")
            lines.append(f"Status: {resource.get('status', 'unknown')}")
            lines.append(f"Performed: {resource.get('performedDateTime', 'unknown')}")

        else:
            # Generic fallback for unsupported resource types
            lines.append(f"Resource ID: {resource.get('id', 'unknown')}")
            lines.append(f"Status: {resource.get('status', 'unknown')}")

        return "\n".join(lines)

    @staticmethod
    def _extract_pdf_text(file_path: Path) -> str:
        """Extract text from a PDF file using PyMuPDF."""
        try:
            import fitz
            doc = fitz.open(str(file_path))
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text.strip()
        except ImportError:
            logger.warning("PyMuPDF not installed — reading as plain text")
            return file_path.read_text(encoding="utf-8", errors="ignore")
