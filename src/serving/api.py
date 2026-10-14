from __future__ import annotations

from dotenv import load_dotenv



load_dotenv()




import logging

import os

import time

import uuid

from datetime import datetime, timezone

from typing import Optional, Any



from fastapi import FastAPI, HTTPException, Depends, Request, status

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel, Field



logger = logging.getLogger(__name__)



# ── App initialization ────────────────────────────────────────────────────────



app = FastAPI(

    title="Healthcare AI Platform API",

    description=(

        "Production-grade healthcare AI platform with RAG pipelines, "

        "multi-agent clinical decision support, treatment recommendation, "

        "and enterprise model serving. Built for Oracle Health AI scale."

    ),

    version="1.0.0",

    docs_url="/docs",

    redoc_url="/redoc",

)



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



# ── Security ──────────────────────────────────────────────────────────────────



security = HTTPBearer(auto_error=False)



VALID_API_KEYS = set(

    filter(None, os.environ.get("API_KEYS", "dev-key-123,test-key-456").split(","))

)





def verify_api_key(

    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),

) -> str:

    """Validate Bearer token API key."""

    if not credentials:

        raise HTTPException(

            status_code=status.HTTP_401_UNAUTHORIZED,

            detail="API key required. Pass as: Authorization: Bearer <api_key>",

        )

    if credentials.credentials not in VALID_API_KEYS:

        raise HTTPException(

            status_code=status.HTTP_403_FORBIDDEN,

            detail="Invalid API key.",

        )

    return credentials.credentials





# ── Rate limiting ─────────────────────────────────────────────────────────────



_rate_limit_store: dict[str, list[float]] = {}

RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))

RATE_LIMIT_WINDOW   = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))





def check_rate_limit(api_key: str = Depends(verify_api_key)) -> str:

    """Enforce per-API-key rate limiting."""

    now = time.time()

    window_start = now - RATE_LIMIT_WINDOW

    requests = _rate_limit_store.get(api_key, [])

    requests = [r for r in requests if r > window_start]



    if len(requests) >= RATE_LIMIT_REQUESTS:

        raise HTTPException(

            status_code=status.HTTP_429_TOO_MANY_REQUESTS,

            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s.",

        )



    requests.append(now)

    _rate_limit_store[api_key] = requests

    return api_key





# ── Request/Response models ───────────────────────────────────────────────────



class BaseRequest(BaseModel):

    request_id: Optional[str] = Field(

        default_factory=lambda: str(uuid.uuid4()),

        description="Optional client request ID for tracing",

    )



class BaseResponse(BaseModel):

    request_id: str

    timestamp: str = Field(

        default_factory=lambda: datetime.now(timezone.utc).isoformat()

    )

    processing_time_ms: Optional[float] = None



class RAGQueryRequest(BaseRequest):

    question: str = Field(..., min_length=5, max_length=2000,

        description="Clinical question to answer")

    patient_id: Optional[str] = Field(None,

        description="Patient ID to filter retrieval")

    n_retrieve: Optional[int] = Field(6, ge=1, le=20,

        description="Number of context chunks to retrieve")



class RAGIngestRequest(BaseRequest):

    text: str = Field(..., min_length=10,

        description="Clinical document text to ingest")

    source: str = Field(..., description="Document source identifier")

    patient_id: Optional[str] = None

    doc_type: Optional[str] = Field("clinical_note",

        description="Document type: clinical_note, guideline, fhir_record")



class RAGQueryResponse(BaseResponse):

    question: str

    answer: str

    sources: list[str]

    chunks_retrieved: int

    model_used: str

    disclaimer: str = (

        "This response is AI-generated decision support. "

        "Clinical decisions must be made by qualified healthcare professionals."

    )



class AgentRequest(BaseRequest):

    query: str = Field(..., min_length=5, max_length=2000,

        description="Clinical question or task for the agent")

    patient_id: Optional[str] = None

    patient_context: Optional[str] = Field(None,

        description="Pre-loaded patient context (optional)")



class AgentResponse(BaseResponse):

    query: str

    patient_id: str

    query_type: str

    final_summary: str

    alerts: list[str]

    drug_safety_findings: str

    diagnostic_findings: str

    treatment_recommendations: str

    disclaimer: str = (

        "AI-generated clinical decision support only. "

        "All recommendations require physician review."

    )



class RecommendationRequest(BaseRequest):

    patient_id: str

    indication: str = Field(...,

        description="Clinical indication: nsclc, heart_failure, type2_diabetes")

    biomarkers: dict[str, Any] = Field(default_factory=dict,

        description="Patient biomarker values")

    egfr: Optional[float] = Field(None, ge=0, le=200)

    performance_status: Optional[int] = Field(None, ge=0, le=4)

    current_medications: list[str] = Field(default_factory=list)

    allergies: list[str] = Field(default_factory=list)

    treatment_line: str = Field("first",

        description="Line of therapy: first, second, third")

    ranking_method: str = Field("ensemble",

        description="Ranking method: weighted_scoring, topsis, ensemble")



class RecommendationResponse(BaseResponse):

    patient_id: str

    indication: str

    n_options: int

    top_recommendations: list[dict]

    disclaimer: str = (

        "Treatment recommendations are AI-generated decision support. "

        "Final treatment decisions require physician judgment and "

        "consideration of individual patient factors."

    )



class DrugSafetyRequest(BaseRequest):

    patient_id: str

    medications: list[str] = Field(..., min_length=1,

        description="Current medication list")

    allergies: list[str] = Field(default_factory=list)

    egfr: Optional[float] = Field(None, ge=0, le=200)

    child_pugh: Optional[str] = Field(None,

        description="Child-Pugh class: A, B, or C")



class DrugSafetyResponse(BaseResponse):

    patient_id: str

    n_medications: int

    n_interactions: int

    n_major_interactions: int

    n_allergy_alerts: int

    critical_alerts: bool

    major_interactions: list[dict]

    allergy_alerts: list[str]

    high_alert_flags: list[str]

    renal_adjustments: list[str]

    polypharmacy_score: int



class SummarizeRequest(BaseRequest):

    clinical_text: str = Field(..., min_length=50,

        description="Clinical document text to summarize")

    patient_id: str

    summary_type: str = Field("sbar",

        description="Summary format: sbar, discharge, medication_rec, handoff, longitudinal")

    additional_context: Optional[str] = None



class SummarizeResponse(BaseResponse):

    patient_id: str

    summary_type: str

    summary_text: str

    word_count: int

    key_alerts: list[str]

    action_items: list[str]





# ── Middleware — request logging ──────────────────────────────────────────────



@app.middleware("http")

async def log_requests(request: Request, call_next):

    start = time.time()

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    logger.info(

        "REQUEST [%s] %s %s",

        request_id, request.method, request.url.path,

    )

    response = await call_next(request)

    duration_ms = (time.time() - start) * 1000

    logger.info(

        "RESPONSE [%s] %d — %.1fms",

        request_id, response.status_code, duration_ms,

    )

    response.headers["X-Request-ID"]      = request_id

    response.headers["X-Processing-Time"] = f"{duration_ms:.1f}ms"

    return response





# ── Health endpoints ──────────────────────────────────────────────────────────



@app.get("/health", tags=["Infrastructure"])

async def health_check():

    """Liveness probe — returns 200 if the service is running."""

    return {

        "status":    "healthy",

        "service":   "healthcare-ai-platform",

        "version":   "1.0.0",

        "timestamp": datetime.now(timezone.utc).isoformat(),

    }





@app.get("/ready", tags=["Infrastructure"])

async def readiness_check():

    """

    Readiness probe — checks all dependencies are available.

    Returns 200 if ready to serve traffic.

    """

    checks = {}



    # Check OpenAI API key

    checks["openai_key"] = bool(os.environ.get("OPENAI_API_KEY"))



    # Check ChromaDB directory

    chroma_dir = os.environ.get("CHROMA_DB_DIR", "./chroma_db")

    checks["chroma_db"] = True



    all_ready = all(checks.values())

    return JSONResponse(

        status_code=200 if all_ready else 503,

        content={

            "status":  "ready" if all_ready else "not_ready",

            "checks":  checks,

            "timestamp": datetime.now(timezone.utc).isoformat(),

        },

    )





# ── RAG endpoints ─────────────────────────────────────────────────────────────



@app.post(

    "/api/v1/rag/query",

    response_model=RAGQueryResponse,

    tags=["RAG Pipeline"],

    summary="Answer a clinical question using RAG",

)

async def rag_query(

    request: RAGQueryRequest,

    api_key: str = Depends(check_rate_limit),

):

    """

    Answer a clinical question using retrieval-augmented generation

    over indexed patient records and clinical guidelines.

    """

    start = time.time()

    try:

        from src.rag.clinical_rag import ClinicalRAGPipeline

        pipeline = ClinicalRAGPipeline(

            api_key=os.environ.get("OPENAI_API_KEY"),

            n_retrieve=request.n_retrieve or 6,

        )

        response = pipeline.query(

            question=request.question,

            patient_id=request.patient_id,

        )

        return RAGQueryResponse(

            request_id=request.request_id,

            question=response.question,

            answer=response.answer,

            sources=response.source_documents,

            chunks_retrieved=response.n_chunks_retrieved,

            model_used=response.model_used,

            processing_time_ms=round((time.time() - start) * 1000, 1),

        )

    except Exception as e:

        logger.error("RAG query error: %s", e)

        raise HTTPException(status_code=500, detail=str(e))





@app.post(

    "/api/v1/rag/ingest",

    tags=["RAG Pipeline"],

    summary="Ingest a clinical document into the vector store",

)

async def rag_ingest(

    request: RAGIngestRequest,

    api_key: str = Depends(check_rate_limit),

):

    """Ingest a clinical document or FHIR resource into the RAG vector store."""

    start = time.time()

    try:

        from src.rag.clinical_rag import ClinicalRAGPipeline

        pipeline = ClinicalRAGPipeline(api_key=os.environ.get("OPENAI_API_KEY"))

        metadata = {

            "source":     request.source,

            "doc_type":   request.doc_type or "clinical_note",

            "patient_id": request.patient_id or "",

        }

        n_chunks = pipeline.ingest_text(request.text, metadata)

        return {

            "request_id":        request.request_id,

            "status":            "ingested",

            "source":            request.source,

            "chunks_created":    n_chunks,

            "processing_time_ms": round((time.time() - start) * 1000, 1),

        }

    except Exception as e:

        logger.error("RAG ingest error: %s", e)

        raise HTTPException(status_code=500, detail=str(e))





# ── Agent endpoint ────────────────────────────────────────────────────────────



@app.post(

    "/api/v1/agents/run",

    response_model=AgentResponse,

    tags=["AI Agents"],

    summary="Run the multi-agent clinical decision support workflow",

)

async def run_agent(

    request: AgentRequest,

    api_key: str = Depends(check_rate_limit),

):

    """

    Run the LangGraph multi-agent clinical decision support workflow.

    Routes through patient context, drug safety, diagnostic, and

    treatment agents based on query type.

    """

    start = time.time()

    try:

        from src.agents.clinical_agent import ClinicalAgentOrchestrator

        agent = ClinicalAgentOrchestrator(

            api_key=os.environ.get("OPENAI_API_KEY"),

        )

        result = agent.run(

            query=request.query,

            patient_id=request.patient_id or "",

            patient_context=request.patient_context or "",

        )

        return AgentResponse(

            request_id=request.request_id,

            query=result["query"],

            patient_id=result["patient_id"],

            query_type=result["query_type"],

            final_summary=result["final_summary"],

            alerts=result["alerts"],

            drug_safety_findings=result["drug_safety_findings"],

            diagnostic_findings=result["diagnostic_findings"],

            treatment_recommendations=result["treatment_recommendations"],

            processing_time_ms=round((time.time() - start) * 1000, 1),

        )

    except Exception as e:

        logger.error("Agent error: %s", e)

        raise HTTPException(status_code=500, detail=str(e))





# ── Recommendation endpoint ───────────────────────────────────────────────────



@app.post(

    "/api/v1/recommend",

    response_model=RecommendationResponse,

    tags=["Treatment Recommendation"],

    summary="Generate biomarker-driven treatment recommendations",

)

async def recommend_treatment(

    request: RecommendationRequest,

    api_key: str = Depends(check_rate_limit),

):

    """

    Generate ranked treatment recommendations based on patient

    biomarker profile, clinical indication, and comorbidities.

    """

    start = time.time()

    try:

        from src.recommender.treatment_recommender import (

            TreatmentRecommender, PatientProfile

        )

        recommender = TreatmentRecommender()

        profile = PatientProfile(

            patient_id=request.patient_id,

            indication=request.indication,

            biomarkers=request.biomarkers,

            egfr=request.egfr,

            performance_status=request.performance_status,

            current_medications=request.current_medications,

            allergies=request.allergies,

            treatment_line=request.treatment_line,

        )

        report = recommender.recommend(profile)

        return RecommendationResponse(

            request_id=request.request_id,

            patient_id=request.patient_id,

            indication=request.indication,

            n_options=len(report.recommendations),

            top_recommendations=[r.to_dict() for r in report.top_n(5)],

            processing_time_ms=round((time.time() - start) * 1000, 1),

        )

    except Exception as e:

        logger.error("Recommendation error: %s", e)

        raise HTTPException(status_code=500, detail=str(e))





# ── Drug safety endpoint ──────────────────────────────────────────────────────



@app.post(

    "/api/v1/drug-safety/screen",

    response_model=DrugSafetyResponse,

    tags=["Drug Safety"],

    summary="Screen a medication list for interactions and safety issues",

)

async def screen_drug_safety(

    request: DrugSafetyRequest,

    api_key: str = Depends(check_rate_limit),

):

    """

    Perform comprehensive medication safety screening including

    drug-drug interactions, allergy contraindications, renal/hepatic

    dose adjustments, and high-alert medication flags.

    """

    start = time.time()

    try:

        from src.agents.drug_interaction_tool import DrugInteractionChecker

        checker = DrugInteractionChecker()

        report = checker.screen_medication_list(

            patient_id=request.patient_id,

            medications=request.medications,

            allergies=request.allergies,

            egfr=request.egfr,

            child_pugh=request.child_pugh,

        )

        summary = report.summary()

        return DrugSafetyResponse(

            request_id=request.request_id,

            patient_id=request.patient_id,

            n_medications=summary["n_medications"],

            n_interactions=summary["n_interactions"],

            n_major_interactions=summary["n_major"],

            n_allergy_alerts=summary["n_allergy_alerts"],

            critical_alerts=summary["critical_alerts"],

            major_interactions=summary["major_interactions"],

            allergy_alerts=summary["allergy_alerts"],

            high_alert_flags=summary["high_alert_flags"],

            renal_adjustments=report.renal_adjustments,

            polypharmacy_score=summary["polypharmacy_score"],

            processing_time_ms=round((time.time() - start) * 1000, 1),

        )

    except Exception as e:

        logger.error("Drug safety error: %s", e)

        raise HTTPException(status_code=500, detail=str(e))





# ── Summarization endpoint ────────────────────────────────────────────────────



@app.post(

    "/api/v1/summarize",

    response_model=SummarizeResponse,

    tags=["Clinical Summarization"],

    summary="Generate a structured clinical summary",

)

async def summarize_clinical_document(

    request: SummarizeRequest,

    api_key: str = Depends(check_rate_limit),

):

    """

    Generate a structured clinical summary in SBAR, discharge,

    medication reconciliation, I-PASS handoff, or longitudinal format.

    """

    start = time.time()

    try:

        from src.agents.summarizer_agent import ClinicalSummarizerAgent

        agent = ClinicalSummarizerAgent(

            api_key=os.environ.get("OPENAI_API_KEY"),

        )

        summary = agent.summarize(

            clinical_text=request.clinical_text,

            patient_id=request.patient_id,

            summary_type=request.summary_type,

            additional_context=request.additional_context,

        )

        return SummarizeResponse(

            request_id=request.request_id,

            patient_id=request.patient_id,

            summary_type=summary.summary_type,

            summary_text=summary.summary_text,

            word_count=summary.word_count,

            key_alerts=summary.key_alerts,

            action_items=summary.action_items,

            processing_time_ms=round((time.time() - start) * 1000, 1),

        )

    except Exception as e:

        logger.error("Summarization error: %s", e)

        raise HTTPException(status_code=500, detail=str(e))





# ── Entry point ───────────────────────────────────────────────────────────────



if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "src.serving.api:app",

        host="0.0.0.0",

        port=int(os.environ.get("PORT", "8000")),

        reload=os.environ.get("ENV", "production") == "development",

        log_level="info",

    )
