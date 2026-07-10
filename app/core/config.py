"""
Central configuration for the Threat Intel RAG assistant.
Reads from environment variables where relevant so secrets never live in code.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- Data paths ---
RAW_CVE_DIR = BASE_DIR / "data" / "raw" / "cve"
RAW_ATTACK_DIR = BASE_DIR / "data" / "raw" / "attack"
RAW_REPORTS_DIR = BASE_DIR / "data" / "raw" / "reports"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# --- NVD (CVE) API ---
NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY = os.getenv("NVD_API_KEY", "")  # optional but raises rate limit 5->50 req/30s

# --- MITRE ATT&CK ---
# Official CTI repo. "enterprise-attack" covers the domain most threat intel questions target.
ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

# --- Embedding model ---
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # BGE-M3 dense vector size

# --- Vector store (Qdrant) ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "threat_intel"

# --- Chunking ---
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai | local
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
