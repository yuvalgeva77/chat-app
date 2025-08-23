from __future__ import annotations
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from app.core.config import *
from app.core.logging_config import get_logger

logger = get_logger("chat-engine")

# Global instances
model = None
tokenizer = None
pipe = None
_histories: Dict[str, List[Dict[str, str]]] = {}


def init_model() -> None:
    """Initialize the model."""
    global model, tokenizer, pipe
    logger.info(f"Loading chat model: {MODEL_NAME}...")

    try:
        # Load tokenizer first
        logger.debug("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Set special tokens early
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device: {device}")

        if torch.cuda.is_available():
            try:
                model = model.to(device)
                logger.debug("Model moved to GPU")
            except Exception as e:
                logger.warning(f"Could not move to GPU, using CPU: {e}")
                device = "cpu"

        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
        )

        logger.info(f"Model {MODEL_NAME} loaded successfully on {device}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise


def _format_prompt_tinyllama(history: List[Dict[str, str]], user_message: str) -> str:
    """Format for TinyLlama chat model."""
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

    # Add recent history
    for msg in history[-4:]:  # Only last 2 exchanges
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    # TinyLlama uses ChatML format
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            pass

    # Fallback format
    prompt = "<|system|>\nYou are a helpful AI assistant.</s>\n"
    for msg in messages[1:]:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt


def _format_prompt_qwen(history: List[Dict[str, str]], user_message: str) -> str:
    """Format for Qwen models."""
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            pass

    # Fallback
    prompt = "System: You are a helpful AI assistant.\n"
    for msg in messages[1:]:
        prompt += f"{msg['role'].title()}: {msg['content']}\n"
    prompt += "Assistant:"
    return prompt


def _format_prompt_smollm(history: List[Dict[str, str]], user_message: str) -> str:
    """Format for SmolLM."""
    messages = []

    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            pass

    # Fallback
    prompt = ""
    for msg in messages:
        prompt += f"<|{msg['role']}|>\n{msg['content']}<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


def _format_prompt_dialogpt(history: List[Dict[str, str]], user_message: str) -> str:
    """Format for DialoGPT - simplified."""
    # For DialoGPT, simpler is better
    conversation = []

    # Only use last few exchanges
    for msg in history[-4:]:
        conversation.append(msg["content"])

    conversation.append(user_message)

    # Join with EOS token
    return tokenizer.eos_token.join(conversation) + tokenizer.eos_token


def _format_prompt(history: List[Dict[str, str]], user_message: str) -> str:
    """Route to appropriate formatter."""
    if MODEL_TYPE == "tinyllama":
        return _format_prompt_tinyllama(history, user_message)
    elif MODEL_TYPE == "qwen":
        return _format_prompt_qwen(history, user_message)
    elif MODEL_TYPE == "smollm":
        return _format_prompt_smollm(history, user_message)
    elif MODEL_TYPE == "dialogpt":
        return _format_prompt_dialogpt(history, user_message)
    else:
        return _format_prompt_tinyllama(history, user_message)  # Default


def _generate_response(prompt: str) -> str:
    """Generate response."""
    try:
        # Calculate input length
        input_tokens = tokenizer.encode(prompt, return_tensors="pt")
        input_length = input_tokens.shape[1]

        logger.info(f"Input length: {input_length} tokens")

        # Generate with careful parameters
        with torch.no_grad():  # Save memory
            outputs = pipe(
                prompt,
                max_new_tokens=min(MAX_NEW_TOKENS, 512 - input_length),  # Ensure we don't exceed context
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                num_return_sequences=1
            )

        if outputs and len(outputs) > 0:
            response = outputs[0]["generated_text"]
            logger.info(f"Raw generated text: '{response}'")

            # If return_full_text didn't work properly
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                logger.info(f"After prompt removal: '{response}'")
        else:
            logger.warning("No output generated")
            return ""

        # Clean response
        response = response

        logger.info(f"Generated response: {response}...")
        return response

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return ""


def _clean_response(response: str) -> str:
    """Clean up response for small models - ensure complete sentences."""
    # Remove common prefixes only if they're at the very start
    prefixes = ["Assistant:", "Bot:", "AI:"]
    for prefix in prefixes:
        if response.strip().startswith(prefix):
            response = response.strip()[len(prefix):].strip()

    # Only stop at actual stop tokens, not colons
    critical_stops = ["<|endoftext|>", "</s>", "<|end|>", "<|user|>"]
    for stop_token in critical_stops:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()

    # Handle line breaks intelligently
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if len(lines) > 1:
        # Only take first line if the second line looks like a new conversation turn
        second_line = lines[1].lower()
        if any(marker in second_line for marker in ["user:", "human:", "assistant:", "bot:"]):
            response = lines[0]
        else:
            # Keep multiple lines if they're part of the same response
            response = '\n'.join(lines[:3])  # Keep up to 3 lines
    elif lines:
        response = lines[0]

    # Try to end at a complete sentence if the response seems cut off
    if response and not response.endswith(('.', '!', '?', '"', "'")):
        # Find the last complete sentence
        sentences = response.split('.')
        if len(sentences) > 1:
            # Take all complete sentences
            complete_sentences = '.'.join(sentences[:-1]) + '.'
            # Only use this if it's substantial
            if len(complete_sentences) > 20:
                response = complete_sentences

    # Remove extra whitespace but preserve structure
    response = ' '.join(response.split())

    return response


def chat_once(session_id: str, user_message: str) -> str:
    """Generate response with small model optimization."""
    global pipe, tokenizer

    if pipe is None or tokenizer is None:
        init_model()

    try:
        if session_id not in _histories:
            _histories[session_id] = []

        logger.info(f"Processing: '{user_message[:50]}...'")

        # Format prompt
        prompt = _format_prompt(_histories[session_id], user_message)

        # Generate response
        reply = _generate_response(prompt)

        # Only fallback if completely empty
        if not reply.strip():
            logger.warning("Empty response, using fallback")
            reply = "Could you rephrase that?"

        # Update history
        _histories[session_id].append({"role": "user", "content": user_message})
        _histories[session_id].append({"role": "assistant", "content": reply})

        # Trim history aggressively for small models
        if len(_histories[session_id]) > HISTORY_MAX_TURNS * 2:
            _histories[session_id] = _histories[session_id][-(HISTORY_MAX_TURNS * 2):]

        logger.info(f"Response: '{reply[:50]}...'")
        return reply

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return "I'm having trouble right now."


def get_history(session_id: str) -> List[Dict[str, str]]:
    return _histories.get(session_id, [])


def reset_history(session_id: Optional[str] = None) -> int:
    if session_id is not None:
        if session_id in _histories:
            del _histories[session_id]
            return 1
        return 0
    else:
        count = len(_histories)
        _histories.clear()
        return count