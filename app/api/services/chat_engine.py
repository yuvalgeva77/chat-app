import os, uuid
from typing import Dict, List, Tuple, Any
import logging
from transformers import pipeline
from app.core.config import *

logger = logging.getLogger("chat-app")

hf_pipe: Any = None
histories: Dict[str, List[Dict[str, str]]] = {}

def init_model() -> None:
    global hf_pipe
    try:
        if LOCAL_MODEL_DIR:
            logger.info(f"Loading LOCAL model from: {LOCAL_MODEL_DIR} (task={PIPELINE_TASK})")
            hf_pipe = pipeline(PIPELINE_TASK, model=LOCAL_MODEL_DIR, local_files_only=True)
        else:
            logger.info(f"Loading model: {MODEL_NAME} (task={PIPELINE_TASK})")
            hf_pipe = pipeline(PIPELINE_TASK, model=MODEL_NAME)
        _ = hf_pipe("Say hello.")  # warm-up
        logger.info("Model ready")
    except Exception as e:
        if OFFLINE_FALLBACK:
            logger.warning(f"HF model load failed, using offline fallback. Reason: {e}")
            hf_pipe = None
        else:
            raise

def shutdown_model() -> None:
    logger.info("Model shutdown complete")

def _build_prompt(history: List[Dict[str, str]], user_msg: str) -> str:
    lines = ["You are a concise, helpful assistant."]
    trimmed = history[-2 * N_CTX_TURNS :] if len(history) > 2 * N_CTX_TURNS else history
    for t in trimmed:
        lines.append(f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}")
    lines += [f"User: {user_msg}", "Assistant:"]
    return "\n".join(lines)

def _offline_reply(user_msg: str) -> str:
    # Very simple rule-based fallback so the API still works offline
    if any(w in user_msg.lower() for w in ["hi", "hello", "hey"]):
        return "Hi! I’m running in offline mode. Ask me something simple."
    return f"You said: {user_msg}"

def chat_once(message: str, session_id: str | None) -> Tuple[str, str]:
    sid = session_id or str(uuid.uuid4())
    hist = histories.setdefault(sid, [])
    hist.append({"role": "user", "content": message})

    if hf_pipe is None:
        reply = _offline_reply(message)
    else:
        prompt = _build_prompt(hist, message)
        out = hf_pipe(prompt, max_new_tokens=80, temperature=0.7, top_p=0.95)[0]["generated_text"]
        reply = out.strip()

    hist.append({"role": "assistant", "content": reply})
    if len(hist) > 2 * N_CTX_TURNS:
        histories[sid] = hist[-2 * N_CTX_TURNS :]
    return reply, sid

def get_history(session_id: str) -> List[Dict[str, str]]:
    if session_id not in histories:
        raise KeyError("Unknown session")
    return histories[session_id]

def reset_history(session_id: str) -> None:
    if session_id not in histories:
        raise KeyError("Unknown session")
    histories[session_id] = []
    logger.info(f"Session {session_id} reset")
