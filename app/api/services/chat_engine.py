# app/api/services/chat_engine.py
from __future__ import annotations
from typing import Dict, List, Optional, Generator, Tuple
import threading, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.core.config import *
from app.core.logging_config import get_logger
from app.core.rag_index import retrieve as rag_retrieve, get_facts, get_index
from app.core.config import TOP_K_DEFAULT  # for clarity in comments
from app.api.services.response_templates import (
    detect_intent,
    get_template_response,
    build_smart_fallback
)

logger = get_logger("chat-engine")

# ===================== Globals =====================
model = None
tokenizer = None
device = None
_histories: Dict[str, List[Dict[str, str]]] = {}

# ===================== Prompt & Policy =====================
SYSTEM_PROMPT_BASE = (
    "You are “Youval’s Resume Assistant”, a concise, professional assistant for recruiters and hiring teams.\n"
    "RULES:\n"
    "- Answer ONLY using the provided CONTEXT and FACTS below. If the answer isn’t present, say “I don’t know based on my sources.”\n"
    "- Prefer short, scannable answers (bullets). Include short SOURCES by title when relevant.\n"
    "- Never invent numbers, employers, dates, or claims. Do not assume.\n"
    "- Redact phone/email unless `reveal_contact=true` is indicated.\n"
    "- Respond in English only.\n"
    "- Always refer to Youval using female pronouns (she/her).\n"
)

def _use(mode: str) -> bool:
    return str(RESPONSE_STRATEGY).lower() == mode

# ===================== Model init =====================
def init_model() -> None:
    """Initialize model + tokenizer (CPU or CUDA) and warm RAG index."""
    global model, tokenizer, device
    if model is not None and tokenizer is not None:
        return

    logger.info(f"Loading chat model: {MODEL_NAME}.")
    logger.info(f"init_model loaded from: {__file__}")
    logger.info(f"Python executable: {sys.executable}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cpu":
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device)

    mdl.eval()
    if hasattr(mdl.config, "use_cache"):
        mdl.config.use_cache = True
    try:
        if device == "cuda" and hasattr(torch, "compile"):
            mdl = torch.compile(mdl, mode="reduce-overhead")
            logger.info("Model compiled")
    except Exception as e:
        logger.warning(f"torch.compile skipped: {e}")

    model = mdl
    logger.info("Model loaded.")

    # Warm the RAG index so first query isn't slow
    try:
        get_index()
        logger.info("RAG index ready.")
    except Exception as e:
        logger.warning(f"RAG warmup failed (continuing): {e}")


# ===================== Helpers =====================
def _apply_redaction(text: str, reveal: bool) -> str:
    """Mask email/phone unless reveal=True."""
    if reveal:
        return text
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted email]", text)
    s = re.sub(r"\+?\d[\d\s().-]{7,}\d", "[redacted phone]", s)
    return s


def _build_system_prompt(reveal_contact: bool, retrieved_snippets: List[Tuple[str, str]]) -> str:
    facts_json = get_facts() or {}
    reveal_flag = "true" if reveal_contact else "false"
    context_block = "\n\n".join([f"[{title} #{i}] {txt}" for i, (title, txt) in enumerate(retrieved_snippets)])
    extra = (
        "\nWhen asked for contact details, quote ONLY the values present in FACTS JSON. "
        "If a field is missing, say 'Not specified in my sources.' Do not invent."
    )
    return (
        f"{SYSTEM_PROMPT_BASE}\n"
        f"`reveal_contact`={reveal_flag}\n"
        f"{extra}\n"
        f"\nCONTEXT BEGINS\n{context_block}\nCONTEXT ENDS\n"
        f"FACTS (JSON excerpt): {str(facts_json)[:800]}"
    )


def _format_prompt(history: List[Dict[str, str]], user_message: str, reveal_contact: bool,
                   retrieved_snippets: List[Tuple[str, str]]) -> str:
    system_prompt = _build_system_prompt(reveal_contact, retrieved_snippets)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    msgs_to_keep = HISTORY_MAX_TURNS * 2
    for msg in history[-msgs_to_keep:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            logger.debug("apply_chat_template failed; using fallback.")

    prompt = "<|system|>\n" + system_prompt + "</s>\n"
    for msg in messages[1:]:
        tag = "user" if msg["role"] == "user" else "assistant"
        prompt += f"<|{tag}|>\n{msg['content']}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt


def _budget_and_tokenize(prompt: str):
    """
    Ensure we leave room for generation and avoid 'unfeasible length' warnings.
    """
    model_ctx = int(getattr(tokenizer, "model_max_length", 2048) or 2048)
    context_window = max(512, min(model_ctx, 4096))  # Qwen2.5-0.5B ~4096; be conservative

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_window).to(device)
    input_len = int(enc.input_ids.shape[1])

    available = max(0, context_window - input_len - 1)
    max_new = min(MAX_NEW_TOKENS, available)
    if max_new < 8:
        tight_limit = max(512, context_window - 192)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tight_limit).to(device)
        input_len = int(enc.input_ids.shape[1])
        available = max(0, context_window - input_len - 1)
        max_new = max(16, min(MAX_NEW_TOKENS, available))

    min_new = 0
    if MIN_NEW_TOKENS and max_new > 1:
        min_new = min(MIN_NEW_TOKENS, max_new - 1)

    max_length = input_len + max_new  # explicit cap avoids HF warnings

    logger.debug(
        f"[budget] ctx={context_window} input={input_len} max_new={max_new} min_new={min_new} max_length={max_length}")
    return enc, input_len, max_new, min_new, max_length


def _clean_response(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    for marker in ("<|endoftext|>", "</s>", "<|end|>", "<|user|>", "\nUser:", "\nHuman:"):
        if marker in s:
            s = s.split(marker)[0].strip()
    s = " ".join(s.split())
    s = re.sub(r"(?:\s*(?:\d+[\.\)]|[-*•]))\s*$", "", s)
    parts = re.split(r"([.!?])\s+", s)
    if len(parts) >= 3 and len(parts[-1]) < 5:
        s = " ".join(parts[:-2])
    return s.strip() or "I need more context to provide a helpful response."


def _retrieval(query: str, top_k: Optional[int] = None) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    hits = rag_retrieve(query, top_k)
    snippets: List[Tuple[str, str]] = []
    meta: List[Dict] = []
    for (doc, score) in hits:
        snippets.append((doc.title, doc.text))
        meta.append({"doc_id": doc.doc_id, "title": doc.title, "chunk_id": doc.chunk_id, "score": float(score)})
    return snippets, meta


# ---------- Legacy functions kept for backward compatibility ----------
def _bullets(items): return "\n".join(f"- {it}" for it in items if it)


def _facts() -> dict:
    return get_facts() or {}


def _from_facts_contact() -> str:
    b = _facts().get("basics", {}) or {}
    c = (b.get("contact") or {})
    email = c.get("email");
    phone = c.get("phone")
    parts = []
    if email: parts.append(f"Email: {email}")
    if phone: parts.append(f"Phone: {phone}")
    return "\n".join(parts) if parts else "Contact details are not specified in my sources."


def _from_facts_summary() -> str:
    f = _facts();
    b = f.get("basics", {}) or {};
    high = f.get("highlights", []) or []
    lines = [
        f"Name: {b.get('name', 'Youval')}",
        f"Title: {b.get('title')}" if b.get("title") else None,
        f"Location: {b.get('location')}" if b.get("location") else None,
        f"Work Authorization: {b.get('work_authorization')}" if b.get("work_authorization") else None,
        f"Availability: {b.get('availability')}" if b.get("availability") else None,
        "Highlights:",
        _bullets(high) if high else None,
    ]
    return "\n".join([x for x in lines if x])


def _from_facts_availability() -> str:
    avail = (_facts().get("basics", {}) or {}).get("availability")
    return f"Availability: {avail or 'Not specified in my sources.'}"


def _from_facts_auth() -> str:
    auth = (_facts().get("basics", {}) or {}).get("work_authorization")
    return f"Work Authorization: {auth or 'Not specified in my sources.'}"


def _from_facts_location() -> str:
    loc = (_facts().get("basics", {}) or {}).get("location")
    return f"Location: {loc or 'Not specified in my sources.'}"


# ---------- Intent routing (now uses template module) ----------
def _detect_intent(text: str) -> str:
    """Use the enhanced intent detection from templates module"""
    return detect_intent(text)


# ---------- Grounding validator ----------
_WORD_RE = re.compile(r"[A-Za-z0-9@._-]{4,}")


def _is_grounded(reply: str, context_text: str) -> bool:
    """Simple overlap check: if <30% of content words appear in context, treat as ungrounded."""
    if not reply.strip():
        return False
    rep = set(w.lower() for w in _WORD_RE.findall(reply))
    ctx = set(w.lower() for w in _WORD_RE.findall(context_text))
    if not rep:
        return False
    overlap = len(rep & ctx)
    ratio = overlap / max(1, len(rep))
    return ratio >= 0.30


# ---------- Extractive (quote-only) ----------
def _extractive_answer(user_message: str, retrieved_snippets: List[Tuple[str, str]]) -> str:
    """
    Return only sentences from the retrieved context that best answer the query.
    Zero hallucination: we never generate new content.
    """
    import re as _re
    q = (user_message or "").strip().lower()
    sents = []
    for _, txt in retrieved_snippets:
        for s in _re.split(r"(?<=[.!?])\s+", txt or ""):
            s = s.strip()
            if 20 <= len(s) <= 240:
                sents.append(s)
    q_words = set(w for w in _re.findall(r"[a-z0-9]+", q) if len(w) > 2)

    def score(s):
        sw = set(w for w in _re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2)
        return len(sw & q_words)

    best = sorted(sents, key=score, reverse=True)[:6]
    if not best:
        return "I don't know based on my sources."
    seen, out = set(), []
    for s in best:
        sig = " ".join(s.lower().split())[:80]
        if sig in seen: continue
        seen.add(sig)
        out.append(f"- {s}")
    return "\n".join(out) if out else "I don't know based on my sources."


# ===================== Generation =====================
def _generate(prompt: str) -> str:
    try:
        enc, input_len, max_new, min_new, max_length = _budget_and_tokenize(prompt)
        gen_kwargs = {
            **enc,  # Unpack the encoded inputs
            "max_new_tokens": max_new,
            "min_new_tokens": min_new,
            "max_length": max_length,
            "repetition_penalty": REPETITION_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "num_return_sequences": 1,
        }
        if DO_SAMPLE:
            if TEMPERATURE is not None:
                gen_kwargs["temperature"] = TEMPERATURE
            if TOP_K:
                gen_kwargs["top_k"] = TOP_K
            if TOP_P is not None:
                gen_kwargs["top_p"] = TOP_P

        with torch.inference_mode():
            outputs = model.generate(**gen_kwargs)  # Unpack the kwargs properly

        new_tokens = outputs[0][input_len:]
        return _clean_response(tokenizer.decode(new_tokens, skip_special_tokens=True))

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "I encountered an error while generating a response. Please try again."


# ===================== Public API (non-streaming) =====================
def chat_once(
        session_id: str,
        user_message: str,
        reveal_contact: Optional[bool] = None,
        top_k: Optional[int] = None,
) -> Tuple[str, List[Dict]]:
    """
    Non-streaming chat. Returns (reply, sources_meta).
    Deterministic routes for core intents in 'hybrid'; model used for others with grounding check.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        init_model()

    hist = _histories.setdefault(session_id, [])
    hist.append({"role": "user", "content": user_message})
    if HISTORY_TRIM_ON_EACH_REPLY:
        hist[:] = hist[-(HISTORY_MAX_TURNS * 2):]

    intent = _detect_intent(user_message)
    allow_contact = (intent == "contact") or bool(reveal_contact)

    # ---- Strategy: extractive (quote-only) ----
    if _use("extractive"):
        retrieved_snippets, sources_meta = _retrieval(user_message, top_k)
        reply = _extractive_answer(user_message, retrieved_snippets)
        reply = _apply_redaction(reply, allow_contact)
        hist.append({"role": "assistant", "content": reply})
        return reply, sources_meta

    # ---- Strategy: hybrid → deterministic for core recruiter asks ----
    if _use("hybrid"):
        # Use template responses for known intents
        if intent in ["contact", "summary", "experience", "skills", "projects",
                      "availability", "authorization", "location"]:
            import time

            # Add a small delay to simulate thinking (0.5-1.2 seconds)
            thinking_delay = 0.6 + (hash(user_message) % 100) / 100.0 * 0.6  # 0.6-1.2 seconds
            time.sleep(thinking_delay)

            reply = get_template_response(intent)
            reply = _apply_redaction(reply, allow_contact)
            hist.append({"role": "assistant", "content": reply})
            return reply, [{"doc_id": "facts.json", "title": "Facts", "chunk_id": 0, "score": 0.95}]

    # ---- For other asks (or model_only): retrieve + model + grounding ----
    retrieved_snippets, sources_meta = _retrieval(user_message, top_k)
    context_text = "\n\n".join(txt for _, txt in retrieved_snippets)

    prompt = _format_prompt(hist, user_message, allow_contact, retrieved_snippets)
    logger.debug(f"Prompt (truncated): {prompt[:LOG_PROMPT_SNIPPET]}")

    reply = _generate(prompt)
    reply = _apply_redaction(reply, allow_contact)

    if not reply.strip() or not _is_grounded(reply, context_text + "\n" + str(get_facts())):
        # Use smart template-based fallback
        fallback_reply = build_smart_fallback(intent, user_message, retrieved_snippets)
        reply = _apply_redaction(fallback_reply, allow_contact)

    hist.append({"role": "assistant", "content": reply})
    if HISTORY_TRIM_ON_EACH_REPLY:
        hist[:] = hist[-(HISTORY_MAX_TURNS * 2):]

    return reply, sources_meta


# ===================== Public API (streaming) =====================
def chat_stream(
        session_id: str,
        user_message: str,
        reveal_contact: Optional[bool] = None,
        top_k: Optional[int] = None,
) -> Generator[str, None, None]:
    """
    Streaming chat. Yields token chunks; deterministic routes yield once.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        init_model()

    hist = _histories.setdefault(session_id, [])
    hist.append({"role": "user", "content": user_message})
    if HISTORY_TRIM_ON_EACH_REPLY:
        hist[:] = hist[-(HISTORY_MAX_TURNS * 2):]

    intent = _detect_intent(user_message)
    allow_contact = (intent == "contact") or bool(reveal_contact)

    # ---- Strategy: extractive (quote-only) ----
    if _use("extractive"):
        retrieved_snippets, _ = _retrieval(user_message, top_k)
        reply = _extractive_answer(user_message, retrieved_snippets)
        reply = _apply_redaction(reply, allow_contact)
        yield reply
        hist.append({"role": "assistant", "content": reply})
        if HISTORY_TRIM_ON_EACH_REPLY:
            hist[:] = hist[-(HISTORY_MAX_TURNS * 2):]
        return

    # ---- Strategy: hybrid → deterministic for core recruiter asks ----
    if _use("hybrid"):
        # Use template responses for known intents
        if intent in ["contact", "summary", "experience", "skills", "projects",
                      "availability", "authorization", "location"]:
            import time

            # Add a small delay to simulate thinking (0.5-1.5 seconds)
            thinking_delay = 0.8 + (hash(user_message) % 100) / 100.0 * 0.7  # 0.8-1.5 seconds
            time.sleep(thinking_delay)

            reply = get_template_response(intent)
            reply = _apply_redaction(reply, allow_contact)
            yield reply
            hist.append({"role": "assistant", "content": reply})
            if HISTORY_TRIM_ON_EACH_REPLY:
                hist[:] = hist[-(HISTORY_MAX_TURNS * 2):]
            return

    # ---- For other asks (or model_only): retrieve + model + stream ----
    retrieved_snippets, _ = _retrieval(user_message, top_k)
    context_text = "\n\n".join(txt for _, txt in retrieved_snippets)

    prompt = _format_prompt(hist, user_message, allow_contact, retrieved_snippets)

    try:
        enc, input_len, max_new, min_new, max_length = _budget_and_tokenize(prompt)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **enc,  # Unpack the encoded inputs properly
            "max_new_tokens": max_new,
            "min_new_tokens": min_new,
            "max_length": max_length,
            "streamer": streamer,
            "repetition_penalty": REPETITION_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "num_return_sequences": 1,
        }
        if DO_SAMPLE:
            if TEMPERATURE is not None:
                gen_kwargs["temperature"] = TEMPERATURE
            if TOP_K:
                gen_kwargs["top_k"] = TOP_K
            if TOP_P is not None:
                gen_kwargs["top_p"] = TOP_P

        def _run():
            try:
                with torch.inference_mode():
                    model.generate(**gen_kwargs)  # Unpack the kwargs properly
            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
                # Signal the streamer to stop
                streamer.end()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        chunks: List[str] = []
        generation_failed = False

        try:
            for piece in streamer:
                chunks.append(piece)
                yield piece
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            generation_failed = True

        finally:
            full = "".join(chunks).strip()

            # Handle generation failure or ungrounded responses
            if generation_failed or not full or not _is_grounded(full, context_text + "\n" + str(get_facts())):
                # Use smart template-based fallback
                fallback_reply = build_smart_fallback(intent, user_message, retrieved_snippets)
                fb = _apply_redaction(fallback_reply, allow_contact)

                if generation_failed or not full:
                    yield fb
                    cleaned = fb
                    sent_fallback = 1  # Used fallback due to generation failure
                else:
                    cleaned = fb
                    sent_fallback = 2  # Replaced ungrounded response
            else:
                cleaned = _clean_response(full)
                cleaned = _apply_redaction(cleaned, allow_contact)
                sent_fallback = 0

            hist.append({"role": "assistant", "content": cleaned})
            if HISTORY_TRIM_ON_EACH_REPLY:
                hist[:] = hist[-(HISTORY_MAX_TURNS * 2):]

            logger.info(
                f"[stream] done | intent={intent} | contact_allowed={allow_contact} | fallback_mode={sent_fallback}")

    except Exception as e:
        logger.error(f"Chat stream failed: {e}")
        # Fallback to template response
        fallback_reply = build_smart_fallback(intent, user_message, [])
        reply = _apply_redaction(fallback_reply, allow_contact)
        yield reply
        hist.append({"role": "assistant", "content": reply})
        if HISTORY_TRIM_ON_EACH_REPLY:
            hist[:] = hist[-(HISTORY_MAX_TURNS * 2):]


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