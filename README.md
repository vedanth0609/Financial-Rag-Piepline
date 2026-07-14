# FinIntel Core: Advanced Financial RAG Pipeline

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline designed to ingest, process, and analyze multi-page financial documents. The system utilizes hierarchical chunking strategies, handles complex table data structures natively, and couples a **FastAPI** backend microservice with an interactive **Streamlit** financial query interface.

Powered by **LlamaIndex**, **ChromaDB**, and the official modern **Google GenAI production SDK**.

---

## 🚀 System Architecture & Technical Features

Unlike naive RAG implementations, this system implements an advanced hierarchical document indexing technique optimized specifically for complex financial text and tabular financial reports:

*   **Hierarchical Node Structure:** Documents are split into broad parent sections (retaining contextual overview) and granular child sections (retaining specific metrics, numbers, and clauses) to maximize retrieval precision.
*   **Dual-Type Structural Parsing:** Automatically classifies and separates dense paragraphs from financial tables, routing tabular structures to dedicated processing passes.
*   **Decoupled Multi-Tier System:** Built as a standalone REST API backend leveraging FastAPI and an independent data-rendering visualization dashboard using Streamlit.
*   **Vector Isolation Management:** Built-in resilience against OS file-system freezes and path collisions by executing logic entirely outside iCloud/cloud-synced directories.

---

## 🛠️ Production Tech Stack

| Component | Technology | Role / Execution Path |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Custom CSS financial query dashboard, real-time analytics stream |
| **Backend API** | FastAPI / Uvicorn | RESTful endpoints, query engine routing, context management |
| **RAG Orchestration** | LlamaIndex (`llama-index-core`) | Node partitioning, ingestion pipeline, metadata tracking |
| **Vector Database** | ChromaDB (Persistent) | Local high-performance disk storage of vectorized financial embeddings |
| **Embedding Model** | Google GenAI `gemini-embedding-2` | Generates dense vector coordinates |
| **Generation Model** | Google GenAI `gemini-3.5-flash` | Enterprise-tier LLM for synthesis, audit trail generation, and reasoning |

---

## 📁 Repository Structure

```text
financial-rag-pipeline/
├── data/
│   ├── mock/               # Mock data for pipeline verification testing
│   └── production/         
│       └── documents/      # Source financial PDFs/TXTs for production indexing
├── src/
│   ├── app.py              # FastAPI application server & REST routes
│   ├── config.py           # Global settings, provider tokens, and model configurations
│   ├── evaluate.py         # Testing suites and RAG evaluation logic
│   ├── frontend.py         # Streamlit visual query user interface
│   ├── ingestion.py        # Data loading, classification parsing, and ChromaDB sync pipeline
│   └── pipeline.py         # Core retrieval and generation workflow logic
├── .gitignore              # Production security configurations (hides venv, API keys)
├── requirements.txt        # Hardened dependency definitions
└── README.md               # System architectural documentation

```

---

## ⚙️ Local Installation & Configuration

### 1. Prerequisites

Ensure your machine is running Python 3.11+ (Optimized and verified under Python 3.14).

### 2. Environment Setup

Clone the repository and spin up a clean virtual environment to isolate project dependencies:

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/financial-rag-pipeline.git](https://github.com/YOUR_USERNAME/financial-rag-pipeline.git)
cd financial-rag-pipeline

# Initialize and activate the virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install all mandatory production dependencies
pip install -r requirements.txt

```

### 3. Production Variables Configuration

Create a `.env` file in the root directory (this file is excluded from git commits via `.gitignore` to prevent API key leaks):

```env
GOOGLE_API_KEY=AIzaSyYourSecretKeyHere

```

---

## ⚡ Execution Workflow

The application operates in a three-stage workflow: data ingestion, API server initialization, and UI layer boot.

### Step 1: Execute Data Ingestion Pipeline

Place your target financial documents inside `data/production/documents/` and index them into the persistent vector store:

```bash
python src/ingestion.py

```

*The script will parse the items, construct the parent-child relational mappings, compute embeddings via `gemini-embedding-2`, and write to disk.*

### Step 2: Launch FastAPI Backend Microservice

Open a dedicated terminal tab, activate your virtual environment, and spin up the backend engine:

```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload

```

### Step 3: Run the Streamlit UI Dashboard

Open a second terminal tab, activate the virtual environment, and launch the user interface tier:

```bash
streamlit run src/frontend.py

```

The interface will automatically launch in your browser at `http://localhost:8501`.

---

## 🛡️ Production & Architectural Refactor Log

During development, this pipeline underwent extensive optimization and production-hardening phases to align with the latest architectural practices:

1. **Google GenAI SDK Migration:** Migrated completely away from deprecated legacy client libraries (`llama-index-embeddings-gemini` / `llama-index-llms-gemini`) which were locked to the obsolete `v1beta` developer endpoints. Implemented the official modern production `llama-index-llms-google-genai` and `llama-index-embeddings-google-genai` wrappers.
2. **API Model Lifecycle Upgrades:** Upgraded text generation and vectorization targets out of discontinued baseline models (`text-embedding-004`, `gemini-1.5-flash`, and `gemini-2.5-flash`) into active production flagship engines (`gemini-embedding-2` and `gemini-3.5-flash`), eliminating 404 connection routing errors entirely.
3. **UI Content Injection Fix:** Fixed UI parsing constraints by swapping restrictive layout blocks (`st.error`) with structural custom rendering hooks (`st.markdown(..., unsafe_allow_html=True)`), enabling safe HTML styling injections for rich financial data representation.

---

## 📸 System Previews & Engineering Logs

### 1. Development Debugging & SDK Verification

*Engineers tracking down deprecated v1beta client routing traps and verifying successful global refactoring migrations inside the Windsurf workspace:*

### 2. Streamlit UI Layout Bug Resolution

*Catching and resolving layout parsing keyword argument crashes during response delivery visualization passes:*

### 3. Live Production Execution Web Dashboard

*The operational FinIntel Core frontend visualizer processing complex financial data queries live without filesystem or API latency barriers:*

```

```
