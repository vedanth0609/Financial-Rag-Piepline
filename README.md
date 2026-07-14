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
| **Embedding Model** | Google GenAI `gemini-embedding-2` | Generates 1536-dimensional dense vector coordinates |
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
