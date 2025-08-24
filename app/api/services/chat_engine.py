from __future__ import annotations
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.core.config import *
from app.core.logging_config import get_logger

logger = get_logger("chat-engine")

# Global instances
model = None
tokenizer = None
device = None
_histories: Dict[str, List[Dict[str, str]]] = {}


def init_model() -> None:
    """Initialize the model with optimizations."""
    global model, tokenizer, device

    logger.info(f"Loading chat model: {MODEL_NAME}...")

    try:
        # Determine device first
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=True  # Use fast tokenizer when available
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with optimizations (with fallback for missing accelerate)
        model_kwargs = {
            "trust_remote_code": True,
        }

        # Try to use optimizations, but fallback gracefully
        try:
            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                # Try CPU optimizations, but fallback if accelerate is missing
                try:
                    model_kwargs["low_cpu_mem_usage"] = True
                    model_kwargs["torch_dtype"] = torch.float32
                except Exception:
                    logger.warning("Accelerate not available, using standard loading")
                    pass

            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

        except ImportError as e:
            if "accelerate" in str(e).lower():
                logger.warning("Accelerate not available, falling back to standard loading")
                # Fallback to basic loading
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True
                )
            else:
                raise

        # Move model to device if not using device_map
        if device == "cuda" and "device_map" not in model_kwargs:
            model = model.to(device)

        # Set to eval mode for inference
        model.eval()

        # Enable optimizations (with fallbacks)
        if device == "cuda":
            try:
                # Enable memory efficient attention if available
                if hasattr(model.config, 'use_cache'):
                    model.config.use_cache = True
                # Try to compile model for faster inference (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model compiled for optimization")
            except Exception as e:
                logger.warning(f"Advanced optimizations not available: {e}")
        else:
            # Basic CPU optimizations
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True

        logger.info(f"Model loaded successfully on {device}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def _format_prompt(history: List[Dict[str, str]], user_message: str) -> str:
    """Unified prompt formatting with template support."""
    # Build message list
    messages = []

    # Add system message for models that support it
    if MODEL_TYPE in ["tinyllama", "qwen", "smollm"]:
        messages.append({"role": "system", "content": "You are a helpful AI assistant."})

    # Add conversation history (keep it short for efficiency)
    for msg in history[-4:]:  # Only last 2 exchanges
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    # Try to use the tokenizer's chat template first
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            logger.debug("Chat template failed, using fallback")

    # Fallback formatting based on model type
    if MODEL_TYPE == "dialogpt":
        # DialoGPT expects simple concatenation
        conversation = [msg["content"] for msg in history[-4:]] + [user_message]
        return tokenizer.eos_token.join(conversation) + tokenizer.eos_token

    # Default ChatML-style format for other models
    prompt = ""
    if MODEL_TYPE in ["tinyllama", "qwen", "smollm"]:
        prompt = "<|system|>\nYou are a helpful AI assistant.</s>\n"

    for msg in messages[1:] if MODEL_TYPE in ["tinyllama", "qwen", "smollm"] else messages:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}</s>\n"

    prompt += "<|assistant|>\n"
    return prompt


def _generate_response(prompt: str) -> str:
    """Optimized response generation."""
    try:
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # Ensure we don't exceed context
        ).to(device)

        input_length = inputs.input_ids.shape[1]
        max_new = min(MAX_NEW_TOKENS, 512 - input_length - 50)  # Leave buffer

        logger.debug(f"Input length: {input_length}, max_new_tokens: {max_new}")

        # Generate with optimized settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for efficiency
                num_return_sequences=1
            )

        # Decode only the new tokens
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return _clean_response(response)

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return "I'm having trouble generating a response right now."


def _clean_response(response: str) -> str:
    """Clean and format the response."""
    if not response:
        return ""

    # Remove common artifacts
    prefixes_to_remove = ["Assistant:", "Bot:", "AI:", "Response:"]
    for prefix in prefixes_to_remove:
        if response.strip().startswith(prefix):
            response = response.strip()[len(prefix):].strip()

    # Stop at conversation markers
    stop_markers = ["<|endoftext|>", "</s>", "<|end|>", "<|user|>", "\nUser:", "\nHuman:"]
    for marker in stop_markers:
        if marker in response:
            response = response.split(marker)[0].strip()

    # Clean up whitespace and ensure reasonable length
    response = ' '.join(response.split())

    # Try to end at complete sentence if response seems cut off
    if response and len(response) > 20 and not response.endswith(('.', '!', '?', '"', "'")):
        sentences = response.split('.')
        if len(sentences) > 1:
            complete = '.'.join(sentences[:-1]) + '.'
            if len(complete) > 15:  # Only use if substantial
                response = complete

    return response or "I need more context to provide a helpful response."


def chat_once(session_id: str, user_message: str) -> str:
    """Main chat function with optimizations."""
    global model, tokenizer

    if model is None or tokenizer is None:
        init_model()

    try:
        # Initialize history if needed
        if session_id not in _histories:
            _histories[session_id] = []

        logger.info(f"Processing message for session {session_id}")
        # Update conversation history for user
        _histories[session_id].append({"role": "user", "content": user_message})

        # Format prompt
        prompt = _format_prompt(_histories[session_id], user_message)
        # Generate response
        reply = _generate_response(prompt)

        # Update conversation history with response
        _histories[session_id].append({"role": "assistant", "content": reply})

        # Trim history to prevent memory bloat
        max_history = HISTORY_MAX_TURNS * 2  # user + assistant pairs
        if len(_histories[session_id]) > max_history:
            _histories[session_id] = _histories[session_id][-max_history:]

        logger.info(f"Generated response length: {len(reply)} chars")
        return reply

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return "I'm experiencing technical difficulties. Please try again."


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a session."""
    return _histories.get(session_id, [])


def reset_history(session_id: Optional[str] = None) -> int:
    """Reset conversation history."""
    if session_id is not None:
        if session_id in _histories:
            del _histories[session_id]
            return 1
        return 0
    else:
        count = len(_histories)
        _histories.clear()
        return count


def get_stats() -> dict:
    """Get system statistics."""
    return {
        "active_sessions": len(_histories),
        "total_conversations": sum(len(hist) for hist in _histories.values()),
        "device": device,
        "model_name": MODEL_NAME
    }