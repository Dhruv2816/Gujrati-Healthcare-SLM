"""src/config.py — Central configuration loaded from .env"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works regardless of CWD)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# ── Hugging Face ───────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ── Neo4j ─────────────────────────────────────────────────
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "gujarati_health_neo4j")

# ── Redis ─────────────────────────────────────────────────
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "gujarati_health_redis")
REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", 3600))

# ── Paths ─────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"
BOOKS_DIR = DATA_DIR / "books"
CHROMA_DIR = DATA_DIR / "chroma_db_books"
MODELS_DIR = _ROOT / "models"
OUTPUTS_DIR = _ROOT / "outputs"

ADAPTER_PATH = str(MODELS_DIR / "qwen_gu_health_lora")
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
EMBED_MODEL_ID = "intfloat/multilingual-e5-large"
CHROMA_COLLECTION = "medical_books"

# ── Safety ────────────────────────────────────────────────
EMERGENCY_KEYWORDS = [
    "heart attack", "stroke", "unconscious", "bleeding", "poisoning",
    "overdose", "chest pain", "collapse", "ઇમર્જન્સી", "ઝેર", "ગંભીર",
    "emergency", "severe", "breathing difficulty", "not breathing",
]

EMERGENCY_RESPONSE = (
    "⚠️ **તાત્કાલિક:** આ ગંભીર સ્વાસ્થ્ય સ્થિતિ છે. "
    "**108 (Ambulance)** અથવા **112 (Emergency)** ને "
    "અત્યારે જ call કરો. ઘરેલૂ ઉપચાર ન કરો."
)

SYSTEM_PROMPT = (
    "You are a highly safe, accurate Gujarati healthcare assistant. "
    "Answer ONLY in Gujarati. Use ONLY the provided Medical Textbook context. "
    "Do NOT hallucinate. Always end with a safety advisory."
)
