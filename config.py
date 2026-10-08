"""Centralised configuration for V2.

All sensitive values (API keys) and environment-specific paths are loaded from
``.env`` via python-dotenv.  Every other module should import from here rather
than calling os.getenv() directly.

Usage
-----
    from config import REUTERS_DIR, ANALYST_MODEL, OPENAI_API_KEY
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (the directory containing this file)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─────────────────────────── Directory Layout ─────────────────────────────────

DATA_RAW_DIR = os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")
DATA_VECTOR_STORE_DIR = os.getenv("DATA_VECTOR_STORE_DIR", "data/vector_store")
EXPERIMENTS_DIR = os.getenv("EXPERIMENTS_DIR", "experiments/results")

# ─────────────────────────── Corpus Paths ─────────────────────────────────────

REUTERS_DIR = os.getenv(
    "REUTERS_DIR", os.path.join(DATA_RAW_DIR, "reuters")
)
STUDENT_ESSAY_DIR = os.getenv(
    "STUDENT_ESSAY_DIR", os.path.join(DATA_RAW_DIR, "student_essays")
)
PAN2024_DIR = os.getenv(
    "PAN2024_DIR", os.path.join(DATA_RAW_DIR, "pan2024")
)
PAN2023_DIR = os.getenv(
    "PAN2023_DIR", os.path.join(DATA_RAW_DIR, "pan2023")
)
BLOG_AUTHORSHIP_DIR = os.getenv(
    "BLOG_AUTHORSHIP_DIR", os.path.join(DATA_RAW_DIR, "blog_authorship")
)
CEFR_WORDLIST_PATH = os.getenv(
    "CEFR_WORDLIST_PATH", os.path.join(DATA_RAW_DIR, "ENGLISH_CEFR_WORDS.csv")
)

# ─────────────────────────── LLM API Keys ─────────────────────────────────────
# Never hardcode these.  Set them in .env (see .env.example).

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ─────────────────────────── Agent Model Names ────────────────────────────────
# Defaults use gpt-4o for evaluation. Override via .env (e.g. ANALYST_MODEL=gpt-4o-mini)
# for lower-cost development.

ANALYST_MODEL: str = os.getenv("ANALYST_MODEL", "gpt-4o")
SKEPTIC_MODEL: str = os.getenv("SKEPTIC_MODEL", "gpt-4o")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-4o")
AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.2"))

# ─────────────────────────── Retrieval ────────────────────────────────────────

RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
FAISS_INDEX_PATH: str = os.getenv(
    "FAISS_INDEX_PATH", os.path.join(DATA_VECTOR_STORE_DIR, "faiss.index")
)
FAISS_METADATA_PATH: str = os.getenv(
    "FAISS_METADATA_PATH", os.path.join(DATA_VECTOR_STORE_DIR, "metadata.jsonl")
)

# ─────────────────────────── NLP / Feature Extraction ─────────────────────────

SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_md")
GPT2_MODEL: str = os.getenv("GPT2_MODEL", "gpt2")
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# ─────────────────────────── Interpretability ─────────────────────────────────

SHAP_MODEL_PATH: str = os.getenv("SHAP_MODEL_PATH", "models/shap_gbc.joblib")
