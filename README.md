# 🚀 AI Career Copilot

> **A multi-agent, RAG-powered AI system that acts as your personal career mentor — analyzing resumes, detecting skill gaps, generating learning roadmaps, and answering career questions using state-of-the-art LLM technology.**

<br>

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-green?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange?style=flat-square)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [RAG Architecture](#-rag-architecture---retrieval-augmented-generation)
- [LLM Integration Architecture](#-llm-integration-architecture)
- [Multi-Agent Architecture](#-multi-agent-architecture)
- [End-to-End Pipeline](#-end-to-end-ai-pipeline)
- [Workflow Diagrams](#-workflow-diagrams)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Design Decisions](#-design-decisions)
- [Screenshots](#-screenshots)
- [Future Roadmap](#-future-roadmap)

---

## 🧠 Overview

**AI Career Copilot** is not just a chatbot — it is a complete career intelligence system. Built on top of **Retrieval-Augmented Generation (RAG)**, a **multi-agent architecture**, and **Groq's Llama 3.3 70B**, it combines structured document parsing, vector similarity search, and AI reasoning to deliver personalised career guidance.

### The Problem It Solves

| Traditional AI Tools | AI Career Copilot |
|---|---|
| Generic answers | Personalised to your resume + goals |
| No memory | Tracks your progress across sessions |
| No document understanding | Parses resume + JD into structured sections |
| No skill comparison | Exact skill gap analysis with % match score |
| No learning plan | Generates week-by-week roadmap + daily tasks |



## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **Resume vs JD Matching** | Structured parsing + cosine similarity + % match score |
| 🔍 **Skill Gap Detection** | Exact matched/missing skills extracted from both documents |
| 🤖 **RAG Career Chat** | Ask any question — FAISS retrieves context, Groq answers |
| 🗺️ **Career Roadmap** | 4-phase personalised roadmap based on missing skills |
| 📅 **Daily Task Generator** | 7-day hands-on task plan with difficulty levels |
| 📊 **Progress Dashboard** | Charts, skill radar, score tracker, activity log |
| 💾 **Persistent Memory** | User profiles saved across sessions |
| 📥 **Progress Report** | Downloadable JSON progress report |

---

## 🏗️ System Architecture

![RAG Architecture](<Photos & Architecture/premium_system_architecture.png>)
---

## 🔍 RAG Architecture — Retrieval-Augmented Generation

This is the core of the system. Unlike naive RAG that chunks text blindly, this system uses **structured, section-aware parsing** before retrieval.

![RAG Architecture](<Photos & Architecture/RAG_Architecture_image.png>)
### Why This RAG Is Better Than Naive RAG

| Naive RAG | This System |
|---|---|
| Split by `\n\n` blindly | Section-aware parsing (Skills, Experience, etc.) |
| All chunks equal weight | Source-filtered retrieval (resume vs JD separately) |
| No relevance filter | L2 threshold — irrelevant results dropped |
| Raw text shown to user | Structured labels: `Resume — Skills`, `JD — Required` |
| Just retrieves | Retrieves **AND** compares with skill gap analysis |

---

## 🤖 LLM Integration Architecture

```
┌───────────────────────────────────────────────────────┐
│              PROMPT BUILDER                            │
│                                                        │
│  Inputs combined:                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ User Profile│  │  RAG Chunks  │  │  Skill Gap   │ │
│  │ goal        │  │  (top-k      │  │  Analysis    │ │
│  │ experience  │  │   results)   │  │  matched     │ │
│  │ skills      │  │              │  │  missing     │ │
│  │ score       │  │              │  │  score       │ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘ │
│         └────────────────┴──────────────────┘         │
│                          │                             │
│              ┌───────────▼───────────┐                │
│              │   STRUCTURED PROMPT   │                │
│              │  System role          │                │
│              │  User context         │                │
│              │  Retrieved docs       │                │
│              │  Skill gap data       │                │
│              │  User question        │                │
│              │  Instructions         │                │
│              └───────────┬───────────┘                │
└──────────────────────────┼────────────────────────────┘
                           │ HTTPS API call
┌──────────────────────────▼────────────────────────────┐
│              GROQ INFERENCE ENGINE                     │
│                                                        │
│  Model: llama-3.3-70b-versatile                       │
│  Speed: ~500 tokens/second (fastest free LLM)         │
│  Context: 128K tokens                                  │
│  Free tier: 1,000 requests/day                         │
│                                                        │
│  Parameters used:                                      │
│  temperature = 0.7 (creative but controlled)           │
│  max_tokens  = 1024 (career advice length)             │
└──────────────────────────┬────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   CAREER MENTOR RESPONSE│
              │  Personalised advice    │
              │  Specific skill gaps    │
              │  Free resource links    │
              │  Encouragement          │
              └─────────────────────────┘
```

---

## 🧩 Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────┐
│                 AGENT ORCHESTRATION                  │
│                    (app.py)                          │
└────┬──────────────┬──────────────┬──────────────────┘
     │              │              │
┌────▼────┐   ┌─────▼─────┐  ┌───▼──────────────────┐
│ ADVISOR │   │  PLANNER  │  │   TASK GENERATOR      │
│  AGENT  │   │   AGENT   │  │       AGENT           │
│         │   │           │  │                        │
│ Analyzes│   │ Generates │  │ Creates 7-day plan    │
│ skill   │   │ 4-phase   │  │ Difficulty levels     │
│ gap     │   │ career    │  │ Skill-focused tasks   │
│         │   │ roadmap   │  │ Mark done + track pts │
│ Returns:│   │           │  │                        │
│ matched │   │ Returns:  │  │ Returns:               │
│ missing │   │ phases    │  │ task list per day     │
│ score % │   │ duration  │  │ estimated hours       │
│ verdict │   │ focus     │  │ completion status     │
│ advice  │   │           │  │                        │
└────┬────┘   └─────┬─────┘  └───┬──────────────────┘
     │              │              │
     └──────────────┴──────────────┘
                    │
        ┌───────────▼────────────┐
        │    PROMPT BUILDER      │
        │  Combines all context  │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │     LLM ENGINE         │
        │  (Groq / Llama 3.3)    │
        └────────────────────────┘
```

---

## 🔄 End-to-End AI Pipeline

```
USER INPUT
    │
    ▼
┌─────────────────────────────────────┐
│  1. PROFILE SETUP                    │
│     username, goal, skills, level    │
│     → saved to memory (JSON/DB)      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  2. DOCUMENT INGESTION               │
│     Resume text + JD text           │
│     → Parser → structured sections  │
│     → Chunker → labelled chunks     │
│     → Embedder → 384-dim vectors    │
│     → FAISS → indexed + saved       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  3. COMPARISON LAYER                 │
│     extract_skills(resume) →set A   │
│     extract_skills(jd)     →set B   │
│     matched = A ∩ B                 │
│     missing = B - A                 │
│     score   = |matched|/|B| × 100  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  4. RETRIEVAL (at query time)        │
│     query → embed → L2 search       │
│     threshold filter → top-k chunks │
│     source filter → resume or JD    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  5. PROMPT CONSTRUCTION              │
│     profile + chunks + gap data     │
│     → structured prompt template    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  6. LLM INFERENCE (Groq)             │
│     Llama 3.3 70B reads full context│
│     generates personalised response │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  7. RESPONSE + PROGRESS UPDATE       │
│     UI displays structured output   │
│     progress saved (score + tasks)  │
│     session tracked in memory       │
└─────────────────────────────────────┘
```

---

## 📊 Workflow Diagrams

### Resume–Job Matching Workflow
```
Resume Text ──┐
              ├─→ Parser ─→ Skills extracted ─→ Set A (resume skills)
JD Text ──────┘          └─→ Skills extracted ─→ Set B (JD skills)
                                                        │
                                    matched = A ∩ B ◄──┘
                                    missing = B - A
                                    score   = |A∩B| / |B| × 100
                                                        │
                                    Advisor Agent verdict
                                    Strong / Moderate / Needs work
```

### Query Processing Workflow
```
User types question
      │
      ▼
embed_text(query) → 384-dim vector
      │
      ▼
FAISS.search(query_vector, k=12)
      │
      ▼
Filter: distance < threshold (2.0)
      │
      ▼
Filter: source = resume / jd / both
      │
      ▼
Deduplicate by text prefix
      │
      ▼
Return top-k structured results
{label, section, text, source, score}
```

### Embedding & Indexing Workflow
```
Raw text chunks
      │
      ▼
SentenceTransformer("all-MiniLM-L6-v2")
      │
      ▼
encode(texts) → numpy float32 array
shape: (N, 384)
      │
      ▼
faiss.IndexFlatL2(384)
index.add(vectors)
      │
      ▼
faiss.write_index(index, path)
json.dump(chunks, path)   ← metadata saved alongside
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **UI** | Streamlit 1.50 | Fastest Python UI framework, no frontend code needed |
| **Embeddings** | `all-MiniLM-L6-v2` | 384-dim, 90MB, best speed/quality balance for semantic search |
| **Vector DB** | FAISS (IndexFlatL2) | Facebook's production-grade exact similarity search |
| **LLM** | Llama 3.3 70B via Groq | Fastest inference (500 tok/s), free tier, 128K context |
| **Parsing** | Custom section parser | Section-aware chunking — unique approach vs naive splitting |
| **Memory** | JSON / Supabase | Local dev: JSON. Production: Supabase free PostgreSQL |
| **Charts** | Plotly | Interactive gauge, radar, bar, line charts |
| **Secrets** | python-dotenv | Local `.env` + Streamlit secrets for deployment |

---

## 📁 Project Structure

```
ai-career-copilot/
│
├── app.py                      ← Main Streamlit app (6 pages)
├── requirements.txt            ← All pip packages
├── .env                        ← API keys (never committed)
├── .gitignore                  ← Protects secrets + venv
│
├── rag/                        ← RAG pipeline
│   ├── __init__.py
│   ├── parser.py               ← Structured document parser
│   ├── embedder.py             ← Sentence transformer wrapper
│   ├── vector_store.py         ← FAISS build + load
│   └── retriever.py            ← Query + comparison logic
│
├── agents/                     ← Multi-agent system
│   ├── __init__.py
│   ├── advisor.py              ← Skill gap analysis agent
│   ├── planner.py              ← Career roadmap agent
│   ├── task_generator.py       ← Daily task generation agent
│   ├── prompt_builder.py       ← Structured prompt templates
│   └── llm_engine.py           ← Groq API wrapper
│
├── memory/                     ← Persistent memory layer
│   ├── __init__.py
│   └── user_memory.py          ← User profile + progress storage
│
└── data/                       ← Local storage (gitignored)
    ├── users.json
    ├── plans.json
    ├── tasks.json
    └── faiss_index/
        ├── index.faiss
        └── chunks.json
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- VS Code
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-career-copilot.git
cd ai-career-copilot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

> 🔑 Get your **free** Groq API key at [console.groq.com](https://console.groq.com) — no credit card required

### 5. Run the Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔐 API Key Security

**Your API keys are NEVER stored in this repository.**

```
.env                    ← gitignored — your local secrets
.streamlit/secrets.toml ← gitignored — deployment secrets
```

If you fork this repo, you must:
1. Create your own `.env` file locally
2. Add your own Groq API key
3. Never commit `.env` to Git

---

## 🧠 Design Decisions

### Why FAISS over ChromaDB or Pinecone?
- **Zero latency**: Runs in-process, no network call to vector DB
- **Exact search**: IndexFlatL2 gives guaranteed nearest neighbours
- **Offline capable**: No external service dependency
- **Lightweight**: Single `.faiss` file, easily saved and loaded

### Why `all-MiniLM-L6-v2` over larger models?
- **Speed**: 5x faster than `all-mpnet-base-v2` with only 3% quality drop
- **Size**: 90MB vs 420MB — fits in free-tier RAM
- **Proven**: State-of-the-art for semantic similarity on SBERT benchmarks
- **Dimensionality**: 384-dim hits the sweet spot for FAISS search speed

### Why Groq over OpenAI or Gemini?
- **Free**: 1,000 requests/day, no credit card
- **Fastest**: ~500 tokens/second inference (10x OpenAI speed)
- **Quality**: Llama 3.3 70B matches GPT-4 on reasoning benchmarks
- **Reliability**: No quota surprises unlike Gemini free tier

### Why section-aware parsing over naive chunking?
- Prevents mixing resume experience with JD requirements
- Enables source-filtered retrieval (resume-only vs JD-only queries)
- Produces labelled results users can interpret immediately
- Enables the comparison layer (matched vs missing skills)

### Why threshold filtering (L2 distance < 2.0)?
- Naive RAG returns results even when query is irrelevant
- Threshold ensures only semantically similar chunks surface
- Eliminates the "Results 1-4 show unrelated content" problem seen in early testing

---

## 📸 Screenshots

> *Add screenshots of your running app here*

| Resume Analyzer | Match Score | AI Chat |
|---|---|---|
| ![Analyser](<Photos & Architecture/Analyzer.png>) | ![Match Score](<Photos & Architecture/Match Score.png>) |  ![AI Chat](<Photos & Architecture/AI Chat.png>) |

| Career Roadmap | Daily Tasks | Dashboard |
|---|---|---|
| ![Roadmap](<Photos & Architecture/Career Roadmap.png>) |![Daily Tasks](<Photos & Architecture/Career Roadmap.png>)|![Dashboard](<Photos & Architecture/Dashboard.png>) |

---

## 🔮 Future Roadmap

| Phase | Feature | Status |
|---|---|---|
| ✅ Phase 1 | Basic RAG + Streamlit UI | Complete |
| ✅ Phase 2 | Persistent memory (user profiles) | Complete |
| ✅ Phase 3 | Multi-agent system | Complete |
| ✅ Phase 4 | LLM integration (Groq) | Complete |
| ✅ Phase 5 | Full dashboard + charts | Complete |
| 🔄 Phase 6 | Cloud deployment | In progress |
| 📋 Phase 7 | Resume PDF upload + parsing | Planned |
| 📋 Phase 8 | Market trends integration | Planned |
| 📋 Phase 9 | Claude API for production | Planned |
| 📋 Phase 10 | Multi-user with authentication | Planned |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.© 2026 Gege

---

## 👨‍💻 Author

**GEDENDHAR SIVAKUMAR**  
[![LinkedIn](https://www.linkedin.com/in/gedendhar-s-23b22a197/)

---

## ⭐ Acknowledgements

- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Research
- [Sentence Transformers](https://www.sbert.net/) — UKP Lab
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [Streamlit](https://streamlit.io) — The fastest way to build ML apps
- [Meta Llama](https://llama.meta.com) — Open source LLM

---

<p align="center">
  <b>Built with ❤️ using RAG + Multi-Agent AI + Groq</b><br>
  <i>From resume to roadmap — AI-powered career guidance</i>
</p>
