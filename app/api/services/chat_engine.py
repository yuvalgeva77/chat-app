from __future__ import annotations
from typing import Dict, List, Literal
from transformers import AutoTokenizer, pipeline
from app.core.config import *
from app.core.logging_config import *

logger = get_logger("chat-app")

hf_pipe = None
tokenizer: AutoTokenizer | None = None
Role = Literal["user", "assistant"]

# In-memory history: session_id -> list[{"role": "user"|"assistant", "content": str}]
_histories: Dict[str, List[Dict[str, str]]] = {}


def init_model() -> None:
    """Load tokenizer + pipeline and set sane padding for GPT2-family models."""
    global hf_pipe, tokenizer
    logger.info("Loading model: %s (task=%s)", MODEL_NAME, PIPELINE_TASK)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        # GPT2 & DialoGPT don't define pad; use EOS to silence warnings and allow batching
        tokenizer.pad_token_id = tokenizer.eos_token_id
    hf_pipe = pipeline(PIPELINE_TASK, model=MODEL_NAME, tokenizer=tokenizer)
    _ = hf_pipe("Hello")  # warm-up
    logger.info("Model ready")


def shutdown_model() -> None:
    """Clean up model resources."""
    global hf_pipe, tokenizer
    if hf_pipe is not None:
        logger.info("Shutting down model")
        hf_pipe = None
        tokenizer = None
        logger.info("Model shutdown complete")


def _trim_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if HISTORY_MAX_TURNS <= 0:
        return history
    return history[-(HISTORY_MAX_TURNS * 2):]  # keep last N pairs


def _build_prompt(session_id: str, user_msg: str) -> str:
    """Build a clean prompt the model can follow."""
    turns = _histories.get(session_id, [])
    turns = _trim_history(turns)

    lines: List[str] = [SYSTEM_PROMPT]
    for t in turns:
        prefix = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {t['content']}")
    lines.append(f"User: {user_msg}")
    lines.append("Assistant:")  # important: steer generation
    return "\n".join(lines)


def _postprocess_reply(raw_generated: str, prompt: str) -> str:
    """
    Remove the prompt echo, cut at next role label, and strip stray labels.
    Works around GPT2-style echoing.
    """
    # 1) Drop the prompt if it's included verbatim
    if raw_generated.startswith(prompt):
        text = raw_generated[len(prompt):]
    else:
        # Fallback: split by the last "Assistant:" and take what follows
        parts = raw_generated.split("Assistant:", 1)
        text = parts[1] if len(parts) == 2 else raw_generated

    # 2) Cut at next role label to avoid the model wandering back to "User:"
    cut_positions = [i for tok in STOP_TOKENS if (i := text.find(tok)) != -1]
    if cut_positions:
        text = text[:min(cut_positions)]

    # 3) Clean stray labels and whitespace
    for label in ("Assistant:", "User:", "System:"):
        text = text.replace(label, "")
    return text.strip()


def chat_once(session_id: str, user_msg: str) -> str:
    """Generate reply, update history, return reply."""
    if hf_pipe is None:
        raise RuntimeError("Model not initialized")

    if session_id not in _histories:
        _histories[session_id] = []

    prompt = _build_prompt(session_id, user_msg)

    out = hf_pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE if DO_SAMPLE else None,
        top_p=TOP_P if DO_SAMPLE else None,
        repetition_penalty=REPETITION_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        pad_token_id=tokenizer.pad_token_id if tokenizer else None,  # type: ignore[arg-type]
    )

    generated: str = out[0]["generated_text"]
    reply = _postprocess_reply(generated, prompt)

    # Safety fallback if model echoed user exactly or produced empty string
    if not reply or reply.strip().lower() == user_msg.strip().lower():
        reply = "Got it! How can I help you next?"

    # Update history AFTER generation
    _histories[session_id].append({"role": "user", "content": user_msg})
    _histories[session_id].append({"role": "assistant", "content": reply})

    if HISTORY_TRIM_ON_EACH_REPLY:
        _histories[session_id] = _trim_history(_histories[session_id])

    logger.info("chat_once | sid=%s | user='%s' | reply='%s'", session_id, user_msg, reply[:100])
    return reply


# ------- History utilities (requested) -------

def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return the conversation for a session id (empty list if not found)."""
    hist = _histories.get(session_id, [])
    logger.info("get_history | sid=%s entries=%d", session_id, len(hist))
    return hist

def reset_history(session_id: str | None = None) -> int:
    """
    Clear a single session or all sessions.
    Returns number of conversations removed.
    """
    if session_id:
        removed = 1 if session_id in _histories else 0
        _histories.pop(session_id, None)
        logger.warning("reset_history | sid=%s removed=%d", session_id, removed)
        return removed
    count = len(_histories)
    _histories.clear()
    logger.warning("reset_history | cleared_all count=%d", count)
    return count