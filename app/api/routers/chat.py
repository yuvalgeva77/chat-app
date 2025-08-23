from fastapi import APIRouter, HTTPException
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.api.services.chat_engine import chat_once
import logging

router = APIRouter(tags=["chat"])
logger = logging.getLogger()

@router.get("/health")
def health():
    logger.info("Health check called")
    return {"status": "ok"}

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    logger.info(f"Chat request: {req}")
    msg = (req.message or "").strip()
    if not msg:
        logger.error("Empty message")
        raise HTTPException(status_code=400, detail="Empty message.")
    try:
        reply, sid = chat_once(msg, req.session_id)
        return ChatResponse(reply=reply, session_id=sid)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")
