# Optimized model configuration for speed and efficiency

# Model Selection (choose one by uncommenting):
# For best speed/quality balance:
MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_TYPE: str = "tinyllama"

# For smallest memory footprint:
# MODEL_NAME: str = "Qwen/Qwen2-0.5B-Instruct"
# MODEL_TYPE: str = "qwen"

# For best quality (slightly slower):
# MODEL_NAME: str = "HuggingFaceTB/SmolLM-1.7B-Instruct"
# MODEL_TYPE: str = "smollm"

# For ultra-fast responses (lower quality):
# MODEL_NAME: str = "microsoft/DialoGPT-small"
# MODEL_TYPE: str = "dialogpt"

# Optimized generation parameters for speed
MAX_NEW_TOKENS: int = 100  # Reduced for faster responses
DO_SAMPLE: bool = True
TEMPERATURE: float = 0.7  # Slightly lower for more focused responses
TOP_K: int = 30  # Reduced for speed
TOP_P: float = 0.85  # Slightly lower for speed
REPETITION_PENALTY: float = 1.05  # Minimal to avoid over-correction
NO_REPEAT_NGRAM_SIZE: int = 2  # Reduced for speed

# Memory management
HISTORY_MAX_TURNS: int = 4  # Keep recent context
HISTORY_TRIM_ON_EACH_REPLY: bool = True

# Batch processing (if implementing batch endpoints later)
MAX_BATCH_SIZE: int = 4
BATCH_TIMEOUT: float = 0.1  # seconds