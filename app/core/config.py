# Model
MODEL_NAME: str = "microsoft/DialoGPT-small"
PIPELINE_TASK: str = "text-generation"

# Conversation policy
SYSTEM_PROMPT: str = "You are a concise, helpful assistant."
HISTORY_MAX_TURNS: int = 5
HISTORY_TRIM_ON_EACH_REPLY: bool = True

# Generation defaults tuned for DialoGPT
MAX_NEW_TOKENS: int = 60
DO_SAMPLE: bool = True
TEMPERATURE: float = 0.8
TOP_P: float = 0.9
REPETITION_PENALTY: float = 1.15
NO_REPEAT_NGRAM_SIZE: int = 3

# Text cut markers (client-side stop)
STOP_TOKENS: list[str] = ["\nUser:", "\nAssistant:", "\nSystem:"]
