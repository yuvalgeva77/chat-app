# Optimized model configuration for speed, relevance, and stability

# Model Selection (choose one by uncommenting):
# For best speed/quality balance:
MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_TYPE: str = "tinyllama"

# For smallest memory footprint (CPU-friendly):
# MODEL_NAME: str = "Qwen/Qwen2-0.5B-Instruct"
# MODEL_TYPE: str = "qwen"

# For best quality (slightly slower):
# MODEL_NAME: str = "HuggingFaceTB/SmolLM-1.7B-Instruct"
# MODEL_TYPE: str = "smollm"

# For ultra-fast responses (lower quality):
# MODEL_NAME: str = "microsoft/DialoGPT-small"
# MODEL_TYPE: str = "dialogpt"

# Generation parameters (near-greedy for on-topic answers)
MAX_NEW_TOKENS: int = 160
DO_SAMPLE: bool = False          # Deterministic decoding by default
TEMPERATURE: float = 0.3         # Only used if DO_SAMPLE=True
TOP_K: int = 0                   # Only used if DO_SAMPLE=True
TOP_P: float = 1.0               # Only used if DO_SAMPLE=True
REPETITION_PENALTY: float = 1.05
NO_REPEAT_NGRAM_SIZE: int = 2

# Memory management
# HISTORY_MAX_TURNS counts "exchanges" (user+assistant pairs) used for prompt building.
HISTORY_MAX_TURNS: int = 2       # Keep last 2 exchanges in the prompt
HISTORY_TRIM_ON_EACH_REPLY: bool = True

# Batch (not used yet)
MAX_BATCH_SIZE: int = 4
BATCH_TIMEOUT: float = 0.1  # seconds
