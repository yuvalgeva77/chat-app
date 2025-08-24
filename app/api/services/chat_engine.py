from __future__ import annotations
from typing import Dict, List, Optional, Generator
import threading
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.core.config import *  # your existing config (MODEL_NAME, MODEL_TYPE, etc.)
from app.core.logging_config import get_logger

logger = get_logger("chat-engine")

# ===== Globals =====
model = None
tokenizer = None
device = None
_histories: Dict[str, List[Dict[str, str]]] = {}


# ===== System Prompt (prevents identity mirroring) =====
SYSTEM_PROMPT = (
    "You are Brainbay's assistant. Be concise, accurate, and on-topic.\n"
    "Rules:\n"
    "1) Never claim to be the user or adopt their name.\n"
    "2) Do not invent personal biography about yourself or the user.\n"
    "3) Use first person only for the assistant role ('I will help with...').\n"
    "4) Prefer short, factual answers unless the user asks for depth.\n"
)

# ===== Init =====
# app/api/services/chat_engine.py

def init_model() -> None:
    """Initialize model + tokenizer without requiring Accelerate."""
    import sys
    global model, tokenizer, device

    logger.info(f"Loading chat model: {MODEL_NAME}.")
    logger.info(f"init_model loaded from: {__file__}")
    logger.info(f"Python executable: {sys.executable}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- CPU path: do NOT pass low_cpu_mem_usage/device_map
    if device == "cpu":
        logger.info("Loading model on CPU (no accelerate flags).")
        model_local = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    else:
        # ---- CUDA path: also avoid device_map unless you intentionally install accelerate
        logger.info("Loading model on CUDA (no device_map).")
        model_local = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model_local = model_local.to(device)

    model_local.eval()

    if hasattr(model_local.config, "use_cache"):
        model_local.config.use_cache = True

    # Optional tiny speed hint on GPU
    try:
        if device == "cuda" and hasattr(torch, "compile"):
            model_local = torch.compile(model_local, mode="reduce-overhead")
            logger.info("Model compiled for optimization")
    except Exception as e:
        logger.warning(f"torch.compile not applied: {e}")

    globals()["model"] = model_local
    logger.info("Model loaded.")

# ===== Prompt Formatting =====
def _format_prompt(history: List[Dict[str, str]], user_message: str) -> str:
    """Format chat prompt. Keep limited history for speed."""
    messages: List[Dict[str, str]] = []

    # Add system
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # Bound history: last N exchanges
    msgs_to_keep = HISTORY_MAX_TURNS * 2  # user + assistant messages
    for msg in history[-msgs_to_keep:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # New user message
    messages.append({"role": "user", "content": user_message})

    # Prefer template
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            logger.debug("apply_chat_template failed; using fallback.")

    # Fallback ChatML-like
    prompt = "<|system|>\n" + SYSTEM_PROMPT + "</s>\n"
    for msg in messages[1:]:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt


def _budget_and_tokenize(prompt: str):
    """Tokenize with a safe context budget."""
    model_ctx = getattr(tokenizer, "model_max_length", 1024) or 1024
    context_window = min(int(model_ctx), 1024)

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_window).to(device)
    input_len = enc.input_ids.shape[1]
    headroom = 64
    max_new = max(16, min(MAX_NEW_TOKENS, context_window - input_len - headroom))
    return enc, input_len, max_new


def _clean_response(text: str) -> str:
    """Clean and format the response, avoiding dangling list markers."""
    if not text:
        return ""

    s = text.strip()

    # Cut at markers
    for marker in ("<|endoftext|>", "</s>", "<|end|>", "<|user|>", "\nUser:", "\nHuman:"):
        if marker in s:
            s = s.split(marker)[0].strip()

    # Compact whitespace
    s = " ".join(s.split())

    # Drop dangling list tokens like "2." / "3)" / "-" / "•"
    s = re.sub(r"(?:\s*(?:\d+[\.\)]|[-*•]))\s*$", "", s)

    # Drop tiny trailing fragment
    parts = re.split(r"([.!?])\s+", s)
    if len(parts) >= 3 and len(parts[-1]) < 5:
        s = " ".join(parts[:-2])

    return s.strip() or "I need more context to provide a helpful response."


# ===== Non-streaming (kept for compatibility) =====
def _generate_response(prompt: str) -> str:
    try:
        enc, input_len, max_new = _budget_and_tokenize(prompt)
        logger.debug(f"Input length: {input_len}, max_new_tokens: {max_new}")

        gen_kwargs = dict(
            **enc,
            max_new_tokens=max_new,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_return_sequences=1,
        )
        # Only pass sampling args if enabled to avoid warnings
        if DO_SAMPLE:
            if TEMPERATURE is not None:
                gen_kwargs["temperature"] = TEMPERATURE
            if TOP_K:
                gen_kwargs["top_k"] = TOP_K
            if TOP_P is not None:
                gen_kwargs["top_p"] = TOP_P

        with torch.inference_mode():
            outputs = model.generate(**gen_kwargs)

        new_tokens = outputs[0][input_len:]
        out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _clean_response(out_text)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "I'm having trouble generating a response right now."


def chat_once(session_id: str, user_message: str) -> str:
    global model, tokenizer
    if model is None or tokenizer is None:
        init_model()

    if session_id not in _histories:
        _histories[session_id] = []

    logger.info(f"Processing message for session {session_id}")

    # Append user and pre-trim
    _histories[session_id].append({"role": "user", "content": user_message})
    if HISTORY_TRIM_ON_EACH_REPLY:
        max_msgs = HISTORY_MAX_TURNS * 2
        if len(_histories[session_id]) > max_msgs:
            _histories[session_id] = _histories[session_id][-max_msgs:]

    prompt = _format_prompt(_histories[session_id], user_message)
    reply = _generate_response(prompt)

    _histories[session_id].append({"role": "assistant", "content": reply})
    max_msgs = HISTORY_MAX_TURNS * 2
    if len(_histories[session_id]) > max_msgs:
        _histories[session_id] = _histories[session_id][-max_msgs:]

    logger.info(f"Generated response length: {len(reply)} chars")
    return reply


# ===== Streaming =====
def chat_stream(session_id: str, user_message: str) -> Generator[str, None, None]:
    """
    Yield tokens as they are generated (for SSE). Appends full cleaned reply to history at the end.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        init_model()

    if session_id not in _histories:
        _histories[session_id] = []

    logger.info(f"[stream] Processing message for session {session_id}")

    # Append user and pre-trim
    _histories[session_id].append({"role": "user", "content": user_message})
    if HISTORY_TRIM_ON_EACH_REPLY:
        max_msgs = HISTORY_MAX_TURNS * 2
        if len(_histories[session_id]) > max_msgs:
            _histories[session_id] = _histories[session_id][-max_msgs:]

    prompt = _format_prompt(_histories[session_id], user_message)
    enc, input_len, max_new = _budget_and_tokenize(prompt)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **enc,
        max_new_tokens=max_new,
        streamer=streamer,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_return_sequences=1,
    )
    if DO_SAMPLE:
        if TEMPERATURE is not None:
            gen_kwargs["temperature"] = TEMPERATURE
        if TOP_K:
            gen_kwargs["top_k"] = TOP_K
        if TOP_P is not None:
            gen_kwargs["top_p"] = TOP_P

    # Run generation in a background thread
    def _run():
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    full_chunks: List[str] = []
    try:
        for piece in streamer:
            full_chunks.append(piece)
            yield piece  # raw piece; client assembles
    finally:
        # When generation ends or client disconnects
        full_text = "".join(full_chunks)
        cleaned = _clean_response(full_text)
        _histories[session_id].append({"role": "assistant", "content": cleaned})
        # Final trim
        max_msgs = HISTORY_MAX_TURNS * 2
        if len(_histories[session_id]) > max_msgs:
            _histories[session_id] = _histories[session_id][-max_msgs:]


# ===== Utilities =====
def get_history(session_id: str) -> List[Dict[str, str]]:
    return _histories.get(session_id, [])


def reset_history(session_id: Optional[str] = None) -> int:
    if session_id is not None:
        return 1 if _histories.pop(session_id, None) is not None else 0
    count = len(_histories)
    _histories.clear()
    return count


def get_stats() -> dict:
    return {
        "active_sessions": len(_histories),
        "total_conversations": sum(len(h) for h in _histories.values()),
        "device": device,
        "model_name": MODEL_NAME,
    }
