# app/core/config.py
from pathlib import Path

# ---------------- Model (public HF, runs locally) ----------------
# Tiny but much cleaner than TinyLlama for this task
MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------- Generation ----------------
MAX_NEW_TOKENS: int = 160
MIN_NEW_TOKENS: int = 24          # ensure we get some output
DO_SAMPLE: bool = False           # deterministic
TEMPERATURE: float | None = 0.3   # used only if DO_SAMPLE=True
TOP_K: int = 0                    # used only if DO_SAMPLE=True
TOP_P: float | None = 1.0         # used only if DO_SAMPLE=True
REPETITION_PENALTY: float = 1.05
NO_REPEAT_NGRAM_SIZE: int = 2

# ---------------- Conversation history ----------------
HISTORY_MAX_TURNS: int = 2
HISTORY_TRIM_ON_EACH_REPLY: bool = True

# ---------------- RAG (local data) ----------------
# These paths work both locally and in Docker due to volume mounting
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR: str = str(BASE_DIR / 'data')          # Points to /app/data in Docker, ./data locally
INDEX_DIR: str = str(BASE_DIR / 'var' / 'index') # Points to /app/var/index in Docker, ./var/index locally

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DEFAULT: int = 3                       # tighter context = fewer glitches
CHUNK_SIZE: int = 600
CHUNK_OVERLAP: int = 80

# ---------------- Behavior strategy ----------------
# "hybrid" (recommended): deterministic routes for core asks, model for the rest
# "model_only": always use the local HF model (with RAG + validator)
# "extractive": quote-only answers from retrieved context (zero hallucination)
RESPONSE_STRATEGY: str = "hybrid"  # ← Changed from "extractive"

# ---------------- Logging ----------------
LOG_PROMPT_SNIPPET: int = 600   # truncate system prompt when logging