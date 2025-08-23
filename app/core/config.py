# Model Configuration
MODEL_NAME: str = "microsoft/DialoGPT-medium"
PIPELINE_TASK: str = "text-generation"

# Conversation settings - Simplified for DialoGPT
SYSTEM_PROMPT: str = """You are a helpful, friendly, and engaging AI assistant. Your goal is to have natural, interesting conversations with users."""

# History settings
HISTORY_MAX_TURNS: int = 5  # Number of conversation turns to keep in history
HISTORY_TRIM_ON_EACH_REPLY: bool = True

# Generation parameters - Optimized for DialoGPT
MAX_NEW_TOKENS: int = 50  # Reduced for more focused responses
DO_SAMPLE: bool = True
TEMPERATURE: float = 0.7  # Controls randomness (0.0 to 1.0)
TOP_P: float = 0.9  # Nucleus sampling parameter
REPETITION_PENALTY: float = 1.2  # Increased to reduce repetition
NO_REPEAT_NGRAM_SIZE: int = 3  # Prevent n-gram repetition

# Text cut markers (client-side stop)
STOP_TOKENS: list[str] = ["\nUser:", "\nAssistant:", "\nSystem:", "<|endoftext|>", "User:", "Assistant:"]