# BEST SMALL MODELS (choose one by uncommenting):

# Option 1: TinyLlama - Modern, fast, good quality (1.1B params)
MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_TYPE: str = "tinyllama"

# Option 2: Qwen2 0.5B - Very small, surprisingly good (0.5B params)
# MODEL_NAME: str = "Qwen/Qwen2-0.5B-Instruct"
# MODEL_TYPE: str = "qwen"

# Option 3: SmolLM - Newest small model (1.7B params)
# MODEL_NAME: str = "HuggingFaceTB/SmolLM-1.7B-Instruct"
# MODEL_TYPE: str = "smollm"

# Option 4: DialoGPT Small - If you want to stick with it (117M params)
# MODEL_NAME: str = "microsoft/DialoGPT-small"
# MODEL_TYPE: str = "dialogpt"

PIPELINE_TASK: str = "text-generation"

# Optimized generation parameters for small models
MAX_NEW_TOKENS: int = 150  # Increased for complete responses
DO_SAMPLE: bool = True
TEMPERATURE: float = 0.8  # Slightly higher for more variety
TOP_K: int = 40
TOP_P: float = 0.9
REPETITION_PENALTY: float = 1.1  # Reduced to allow some natural repetition
NO_REPEAT_NGRAM_SIZE: int = 3

# Keep history shorter for small models
HISTORY_MAX_TURNS: int = 3
HISTORY_TRIM_ON_EACH_REPLY: bool = True

# Stop tokens - only critical ones
STOP_TOKENS: list[str] = [
    "<|endoftext|>", "</s>", "<|end|>", "<|user|>",
    "\nUser:", "\nHuman:", "User:", "Human:"
]