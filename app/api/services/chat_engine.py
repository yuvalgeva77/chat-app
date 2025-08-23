from __future__ import annotations
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from app.core.config import *
from app.core.logging_config import get_logger

logger = get_logger("chat-engine")

# Global model and tokenizer instances
model = None
tokenizer = None
pipe = None

# In-memory conversation history: session_id -> list of messages
_histories: Dict[str, List[Dict[str, str]]] = {}


def init_model() -> None:
    """Initialize the model and tokenizer."""
    global model, tokenizer, pipe

    logger.info(f"Loading model: {MODEL_NAME}")

    # Set device (GPU if available, else CPU)
    device = 0 if torch.cuda.is_available() else -1

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Model {MODEL_NAME} loaded successfully on device: {'GPU' if device == 0 else 'CPU'}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def _format_prompt(history: List[Dict[str, str]], user_message: str) -> str:
    """Format the conversation history and current message into a prompt for DialoGPT."""
    # For DialoGPT, we use a simpler format with conversation flow
    prompt_parts = []

    # Add recent conversation history (keep it shorter to avoid token limits)
    recent_history = history[-6:] if len(history) > 6 else history  # Last 3 exchanges

    for message in recent_history:
        if message["role"] == "user":
            prompt_parts.append(message["content"])
        else:
            prompt_parts.append(message["content"])

    # Add current user message
    prompt_parts.append(user_message)

    # Join with special tokens that DialoGPT expects
    prompt = tokenizer.eos_token.join(prompt_parts) + tokenizer.eos_token

    return prompt


def _generate_response(prompt: str) -> str:
    """Generate a response using the model."""
    try:
        # Get input length to calculate proper max_new_tokens
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        input_length = input_ids.shape[1]

        logger.info(f"Input length: {input_length} tokens")

        # Use max_new_tokens instead of max_length
        response = pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,  # This ensures we generate new tokens beyond input
            min_length=input_length + 10,  # Minimum new tokens
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_k=50,
            top_p=TOP_P,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id
        )

        # Extract and clean the response
        generated_text = response[0]["generated_text"]

        # Remove the input prompt from the response
        reply = generated_text[len(prompt):].strip()

        # Clean up the response
        if tokenizer.eos_token in reply:
            reply = reply.split(tokenizer.eos_token)[0].strip()

        # Additional cleanup for any stop tokens
        for token in STOP_TOKENS:
            if token in reply:
                reply = reply.split(token)[0].strip()

        # Remove any leading/trailing whitespace and newlines
        reply = reply.strip()

        logger.info(f"Generated response length: {len(reply)} chars")
        return reply

    except Exception as e:
        logger.error(f"Error in _generate_response: {str(e)}")
        return ""


def _trim_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Trim the conversation history to the maximum number of turns."""
    if HISTORY_MAX_TURNS <= 0:
        return history
    return history[-(HISTORY_MAX_TURNS * 2):]  # Keep last N pairs


def chat_once(session_id: str, user_message: str) -> str:
    """Generate a response to the user's message."""
    global pipe, tokenizer

    if pipe is None or tokenizer is None:
        init_model()

    try:
        # Initialize session history if it doesn't exist
        if session_id not in _histories:
            _histories[session_id] = []

        logger.info(f"Processing message for session {session_id}: {user_message}")
        logger.info(f"Current history length: {len(_histories[session_id])} messages")

        # Format the prompt with conversation history
        prompt = _format_prompt(_histories[session_id], user_message)
        logger.info(f"Formatted prompt: {prompt[:200]}...")

        # Generate response
        reply = _generate_response(prompt)

        # Fallback responses if generation fails or is too short
        if not reply or len(reply.strip()) < 5:
            logger.warning("Generated reply too short or empty, using fallback")
            if "how are you" in user_message.lower():
                reply = "I'm doing well, thank you for asking! How are you doing today?"
            elif "hello" in user_message.lower() or "hi" in user_message.lower():
                reply = "Hello! It's nice to meet you. What would you like to talk about?"
            elif "what" in user_message.lower() and "name" in user_message.lower():
                reply = "I'm an AI assistant created to help and chat with you. What's your name?"
            else:
                reply = f"That's interesting! Could you tell me more about that?"

        # Update conversation history
        _histories[session_id].append({"role": "user", "content": user_message})
        _histories[session_id].append({"role": "assistant", "content": reply})

        # Trim history if needed
        if HISTORY_TRIM_ON_EACH_REPLY:
            _histories[session_id] = _trim_history(_histories[session_id])

        logger.info(f"Final response: {reply}")
        return reply

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm having trouble generating a response right now. Please try again."


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Get the conversation history for a session."""
    return _histories.get(session_id, [])


def reset_history(session_id: Optional[str] = None) -> int:
    """
    Clear conversation history for a specific session or all sessions.
    Returns the number of sessions cleared.
    """
    if session_id is not None:
        if session_id in _histories:
            del _histories[session_id]
            return 1
        return 0
    else:
        count = len(_histories)
        _histories.clear()
        return count