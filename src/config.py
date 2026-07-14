"""
Configuration module for Financial/Legal RAG evaluation pipeline.
Loads environment variables, validates required API keys, and configures LlamaIndex settings.
"""
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# Load environment variables from .env file
load_dotenv()

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Assert that GEMINI_API_KEY is present
assert GEMINI_API_KEY is not None, (
    "GEMINI_API_KEY environment variable is not set. "
    "Please add your Google AI Studio API key to the .env file."
)
assert GEMINI_API_KEY != "", (
    "GEMINI_API_KEY environment variable is empty. "
    "Please provide a valid Google AI Studio API key in the .env file."
)


def init_llm_and_embeddings():
    """
    Initialize LlamaIndex global settings for cloud-based Gemini embeddings and LLM.
    
    Configures:
    - Gemini cloud-based embeddings using gemini-embedding-2 (instant startup, no local model loading)
    - Gemini 3.5 Flash LLM for generation (uses API key from environment)
    """
    # Configure cloud-based embeddings using Gemini
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-2",
        api_key=GEMINI_API_KEY
    )
    
    # Configure Gemini LLM
    Settings.llm = GoogleGenAI(
        model="gemini-3.5-flash",
        api_key=GEMINI_API_KEY
    )
