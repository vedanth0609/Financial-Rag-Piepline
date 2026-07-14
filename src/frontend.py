"""
Enterprise Financial Intelligence Core Dashboard
Decoupled Streamlit frontend communicating exclusively via HTTP to FastAPI gateway
Bloomberg/FactSet aesthetic with robust error handling and type safety
"""
import os
import requests
import streamlit as st
from typing import Dict, Any, List, Optional

# Read from environment variable in production, fallback to local in dev
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_ENDPOINT = f"{BACKEND_URL}/api/v1/query"

# Page Configuration
st.set_page_config(
    page_title="FinIntel Core",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Theme Styling
def inject_custom_css():
    """Inject professional financial interface theme CSS."""
    custom_css = """
    <style>
    /* Root Variables */
    :root {
        --primary-bg: #0a0e27;
        --secondary-bg: #111827;
        --accent-blue: #1e40af;
        --accent-green: #059669;
        --accent-red: #dc2626;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: #1e293b;
    }
    
    /* Main Container */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #111827 100%);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
    }
    
    /* Headers */
    .main-header {
        background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        padding: 1.5rem 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    /* Status Bar */
    .status-bar {
        background: #1e293b;
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #94a3b8;
        font-size: 0.875rem;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #059669;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Input Containers */
    .stTextInput > div > div > input {
        background: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 0.375rem;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    /* Response Container */
    .response-container {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.375rem;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
    }
    
    .streamlit-expanderContent {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Info/Warning/Error Messages */
    .stAlert {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Blockquote Styling */
    blockquote {
        background: #0f172a;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.375rem;
        color: #94a3b8;
    }
    
    /* Code Blocks */
    code {
        background: #0f172a;
        color: #60a5fa;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
    }
    
    pre {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 1rem;
        overflow-x: auto;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# HTTP Client with Error Handling
def query_backend(query_text: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Query the FastAPI backend with strict error handling.
    Returns parsed JSON response or raises ConnectionError.
    """
    try:
        response = requests.post(
            API_ENDPOINT,
            json={"query": query_text, "top_k": top_k},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise ConnectionError(f"Backend returned status {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to backend server. Please ensure the FastAPI application is running on port 8000.")
    except requests.exceptions.Timeout:
        raise ConnectionError("Backend request timed out. Please try again.")
    except Exception as e:
        raise ConnectionError(f"Unexpected error: {str(e)}")


# Initialize Session State
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_response" not in st.session_state:
    st.session_state.current_response = None
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []


# Main Application
def main():
    inject_custom_css()
    
    # Header Zone
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; color: white; font-size: 2rem;">ENTERPRISE FINANCIAL INTELLIGENCE CORE</h1>
        <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 0.875rem;">Advanced RAG-Powered Financial Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Bar
    st.markdown("""
    <div class="status-bar">
        <div class="status-item">
            <div class="status-dot"></div>
            <span>Engine Status: Active</span>
        </div>
        <div class="status-item">
            <span>🔒 Latency Guard: Enabled</span>
        </div>
        <div class="status-item">
            <span>📡 API Gateway: localhost:8000</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Content Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Interaction Core
        st.subheader("🔍 Financial Query Interface")
        st.markdown('<p style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 1rem;">Analyze balance sheets, cash flows, or core risk vectors...</p>', unsafe_allow_html=True)
        
        # Input Field
        query_input = st.text_input(
            "Enter your financial query:",
            placeholder="e.g., What was OmniCorp's net profit margin in Q4 2025?",
            key="query_input"
        )
        
        # Query Button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            query_button = st.button("🚀 Execute Analysis", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        # Clear Functionality
        if clear_button:
            st.session_state.query_history = []
            st.session_state.current_response = None
            st.session_state.current_sources = []
            st.rerun()
        
        # Query Execution
        if query_button and query_input:
            with st.spinner("Synthesizing multi-page financial nodes and executing vector alignment..."):
                try:
                    result = query_backend(query_input, top_k=5)
                    st.session_state.current_response = result.get("response", "")
                    st.session_state.current_sources = result.get("source_nodes", [])
                    st.session_state.query_history.append({
                        "query": query_input,
                        "response": st.session_state.current_response,
                        "sources": st.session_state.current_sources
                    })
                except ConnectionError as e:
                    error_msg = str(e).lower()
                    # Check if this is a rate-limit error
                    is_rate_limit = any(code in error_msg for code in ['429', 'rate limit', 'quota', 'resource exhausted'])
                    
                    if is_rate_limit:
                        # Show toast notification for rate-limit cooling
                        st.toast("⏳ Rate-limit backoff cooling active. Retrying connection context...", icon="⏱️")
                        st.markdown("""
                        <div style="background: #1e293b; border: 1px solid #f59e0b; border-radius: 0.5rem; padding: 1.5rem; margin: 1rem 0;">
                            <h3 style="color: #f59e0b; margin: 0 0 0.5rem 0;">⏱️ Rate-Limit Cooldown Active</h3>
                            <p style="color: #94a3b8; margin: 0;">The backend is currently in an exponential backoff cooling loop due to API rate limits. Please wait a moment and retry your query.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Standard connection error
                        st.markdown(f"""
                        <div style="background: #1e293b; border: 1px solid #dc2626; border-radius: 0.5rem; padding: 1.5rem; margin: 1rem 0;">
                            <h3 style="color: #dc2626; margin: 0 0 0.5rem 0;">⚠️ Backend Connection Error</h3>
                            <p style="color: #94a3b8; margin: 0;">{str(e)}</p>
                            <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.875rem;">
                                <strong>Action Required:</strong> Please initialize the backend application tier by running:<br>
                                <code style="background: #0f172a; color: #60a5fa; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">uvicorn src.app:app --port 8000</code>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Response Display
        if st.session_state.current_response:
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            st.subheader("📊 Analysis Result")
            st.markdown(st.session_state.current_response)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar - Context Citation Control Center
    with col2:
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
            <h2 style="color: white; margin: 0; font-size: 1.25rem;">📋 AUDIT TRAIL</h2>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.875rem;">Source Citation Control Center</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Row
        st.sidebar.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.sidebar.markdown("**Connection Parameters**")
        st.sidebar.markdown(f"- API Gateway: `localhost:8000`")
        st.sidebar.markdown(f"- Protocol: `HTTP/1.1`")
        st.sidebar.markdown(f"- Timeout: `30s`")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Source Accordions
        if st.session_state.current_sources:
            st.sidebar.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.sidebar.markdown(f"**Retrieved Sources: {len(st.session_state.current_sources)}**")
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            for idx, source in enumerate(st.session_state.current_sources, 1):
                file_name = source.get("file_name", "Unknown")
                page_number = source.get("page_number", "N/A")
                score = source.get("score", 0.0)
                score_percent = f"{score * 100:.1f}%" if score else "N/A"
                text = source.get("text", "")
                
                # Expander Header
                expander_title = f"📄 [Score: {score_percent}] - {file_name} (Page {page_number})"
                
                with st.sidebar.expander(expander_title, expanded=False):
                    # Metadata Header
                    st.markdown(f"""
                    <div style="background: #1e293b; padding: 0.75rem; border-radius: 0.375rem; margin-bottom: 0.75rem; border: 1px solid #334155;">
                        <p style="margin: 0; color: #94a3b8; font-size: 0.75rem;">
                            <strong>Document:</strong> {file_name}<br>
                            <strong>Page:</strong> {page_number}<br>
                            <strong>Similarity:</strong> {score_percent}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Content Body
                    st.markdown(f"""
                    <blockquote>
                        <p style="margin: 0; color: #94a3b8; font-size: 0.875rem; line-height: 1.6;">
                            {text[:500]}{'...' if len(text) > 500 else ''}
                        </p>
                    </blockquote>
                    """, unsafe_allow_html=True)
        else:
            st.sidebar.info("📭 No sources retrieved yet. Execute a query to see citation data.")
        
        # Query History
        if st.session_state.query_history:
            st.sidebar.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.sidebar.markdown(f"**Query History: {len(st.session_state.query_history)}**")
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            for idx, entry in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                with st.sidebar.expander(f"🔍 Query #{idx}", expanded=False):
                    st.markdown(f"**Query:** {entry['query']}")
                    st.markdown(f"**Sources:** {len(entry['sources'])}")


if __name__ == "__main__":
    main()
