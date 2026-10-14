# Healthcare AI Agents

A production-grade healthcare AI platform built for Oracle Health AI scale — featuring RAG pipelines over clinical documents, multi-agent clinical decision support, biomarker-driven treatment recommendation, precision mental health matching, and enterprise FastAPI model serving.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![LangChain](https://img.shields.io/badge/LangChain-Agents%20%7C%20RAG-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-teal)
![FHIR](https://img.shields.io/badge/Standard-FHIR%20R4-orange)

---

## Platform Overview

| Module | Files | Description |
|--------|-------|-------------|
| `src/rag/` | `clinical_rag.py`, `fhir_indexer.py`, `embeddings.py` | RAG pipeline over FHIR R4 records and clinical guidelines |
| `src/agents/` | `clinical_agent.py`, `drug_interaction_tool.py`, `summarizer_agent.py` | LangGraph multi-agent clinical decision support |
| `src/recommender/` | `treatment_recommender.py`, `ranking_engine.py`, `therapist_matcher.py` | Biomarker-driven treatment recommendation and precision mental health matching |
| `src/trial_matching/` | `trial_matcher.py` | Real-time clinical trial matching — mirrors Klinixa platform built at Takeda |
| `src/governance/` | `ai_governance.py` | Bias auditing, fairness metrics, responsible AI checklist, audit logging |
| `src/serving/` | `api.py`, `Dockerfile`, `docker-compose.yml` | FastAPI model serving, health checks, Docker deployment |

---

## Key Features

- **Clinical RAG Pipeline** — Retrieval-augmented generation over FHIR R4 patient records and clinical guidelines using ChromaDB vector store and OpenAI embeddings
- **Multi-Agent CDS** — LangGraph-orchestrated clinical decision support agents for drug interaction checking, contraindication alerts, and patient history summarization
- **Treatment Recommender** — Biomarker-driven treatment ranking combining collaborative filtering, clinical rules, and patient profiles (NSCLC, Heart Failure, T2DM)
- **Therapist-Patient Matching** — Precision mental healthcare matching engine using PHQ-9/GAD-7 clinical profiles, patient preferences, and historical outcome data
- **AI Governance** — Bias auditing across protected demographic attributes, immutable audit logging, pre-deployment responsible AI checklist, NIST AI RMF alignment
- **Enterprise Serving** — FastAPI REST API with JWT auth, rate limiting, health checks, model versioning, and Docker/OCI deployment
- **AI Transformation Strategy** — Full enterprise AI roadmap in `STRATEGY.md` covering governance, ROI measurement, pilot sequencing, and scaling

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| LLM / RAG | openai, langchain, langchain-community, chromadb, tiktoken |
| Agents | langgraph, langchain-core, openai tools |
| ML / DL | scikit-learn, TensorFlow, numpy, pandas |
| FHIR | fhir.resources, requests |
| Governance | Custom fairness framework, NIST AI RMF alignment |
| Serving | FastAPI, uvicorn, pydantic, docker |
| Testing | pytest, pytest-asyncio, httpx |

---

## Getting Started

```bash
git clone https://github.com/girish-rajeev/healthcare-ai-agents.git
cd healthcare-ai-agents
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
uvicorn src.serving.api:app --reload
```

API documentation available at `http://localhost:8000/docs` after startup.

---

## Architecture
FHIR R4 Records / Clinical Guidelines / EHR Notes
|
v
RAG Pipeline           --  Embed · index · retrieve · generate (GPT-4o)
|
v
AI Agent Layer         --  Drug safety · diagnostic · treatment · summarization
|
v
Recommendation Engine  --  Treatment ranking · therapist-patient matching
|
v
AI Governance          --  Bias auditing · audit logging · responsible AI checks
|
v
FastAPI Serving        --  REST API · JWT auth · Docker · OCI-ready
|
v
Clinician / Member Facing Applications

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/rag/query` | Answer clinical questions using RAG |
| `POST` | `/api/v1/rag/ingest` | Ingest clinical documents into vector store |
| `POST` | `/api/v1/agents/run` | Run multi-agent clinical decision support |
| `POST` | `/api/v1/recommend` | Generate treatment recommendations |
| `POST` | `/api/v1/drug-safety/screen` | Screen medication list for interactions |
| `POST` | `/api/v1/summarize` | Generate SBAR/discharge/handoff summary |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe |

---

## Repository Structure

```
healthcare-ai-agents/
|-- .github/workflows/ci.yml       GitHub Actions CI pipeline
|-- src/
|   |-- rag/
|   |   |-- clinical_rag.py        RAG pipeline over FHIR records
|   |   |-- fhir_indexer.py        FHIR R4 bulk indexer
|   |   +-- embeddings.py          Clinical embedder with caching
|   |-- agents/
|   |   |-- clinical_agent.py      LangGraph multi-agent CDS
|   |   |-- drug_interaction_tool.py  Drug safety checker
|   |   +-- summarizer_agent.py    SBAR/discharge summarizer
|   |-- recommender/
|   |   |-- treatment_recommender.py  Biomarker-driven recommender
|   |   |-- ranking_engine.py      TOPSIS + ensemble ranker
|   |   +-- therapist_matcher.py   Precision mental health matching
|   |-- trial_matching/
|   |   +-- trial_matcher.py       Real-time trial matching (Klinixa)
|   |-- governance/
|   |   +-- ai_governance.py       Bias auditing + audit logging
|   +-- serving/
|       |-- api.py                 FastAPI REST endpoints
|       |-- Dockerfile
|       +-- docker-compose.yml
|-- tests/
|   +-- test_healthcare_ai.py
|-- notebooks/
|-- data/
|-- STRATEGY.md                    Enterprise AI transformation roadmap
|-- LEADERSHIP.md                  Team structure, OKRs, hiring philosophy
|-- requirements.txt
+-- README.md
```
---

## Regulatory and Clinical Standards

| Standard | Purpose |
|----------|---------|
| HL7 FHIR R4 | Patient data interoperability |
| SNOMED CT | Clinical terminology |
| RxNorm | Drug terminology and normalization |
| ICD-10-CM | Diagnosis coding |
| MedDRA v26.0 | Adverse event terminology |
| FDA Drug Interaction Guidance | Drug safety screening |
| NIST AI RMF | AI governance and risk management |
| HIPAA Privacy and Security Rules | Patient data protection |

---

## AI Governance

This platform implements a comprehensive responsible AI framework:

- **Bias auditing** — Fairness metrics across gender, race/ethnicity, age, language, and insurance type
- **Audit logging** — Immutable SHA-256 hashed decision records for every AI-assisted clinical decision
- **Pre-deployment checklist** — 12-item responsible AI checklist enforced before any model goes to production
- **Crisis safety** — Hard-coded escalation paths for mental health crisis detection — no exceptions
- **HIPAA compliance** — PHI de-identification enforced before all LLM calls
- **Human oversight** — All AI outputs are decision support only — final decisions rest with clinicians

See `STRATEGY.md` for the full enterprise AI transformation roadmap and governance framework.

---

## Author

**Girish Rajeev**

Clinical Data Scientist | Data Analyst | Regulatory Standards Leader | AI/ML Solution Engineer

- LinkedIn: [linkedin.com/in/girish-rajeev-756808138](https://www.linkedin.com/in/girish-rajeev-756808138/)
- GitHub: [github.com/girish-rajeev](https://github.com/girish-rajeev)

---

*Part of a three-repository clinical AI portfolio:*
- [clinical-data-pipeline](https://github.com/girish-rajeev/clinical-data-pipeline) — SDTM/ADaM/TLF pipeline with LLM automation
- [regulatory-submissions-pipeline](https://github.com/girish-rajeev/regulatory-submissions-pipeline) — eCTD/IND/NDA/BLA regulatory automation
- [healthcare-ai-agents](https://github.com/girish-rajeev/healthcare-ai-agents) — RAG, agents, recommender, governance, serving
