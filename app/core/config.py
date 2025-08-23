import os
MODEL_NAME = os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2")
PIPELINE_TASK = os.getenv("PIPELINE_TASK", "text-generation")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "")  # e.g. "models/tiny-gpt2"
N_CTX_TURNS = int(os.getenv("N_CTX_TURNS", "6"))
OFFLINE_FALLBACK = os.getenv("OFFLINE_FALLBACK", "0") == "1"
