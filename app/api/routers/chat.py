from uuid import uuid4
from fastapi import APIRouter, HTTPException, Query, Request
from app.api.schemas.chat import *
from app.api.services.chat_engine import *
from app.core.logging_config import get_logger

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)

@router.get("/health")
def health() -> dict:
    logger.info("Health check endpoint called")
    return {"status": "ok"}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Chat request from {client_host}: {chat_request}")
    
    session_id = chat_request.session_id or str(uuid4())
    try:
        logger.debug(f"Processing message for session {session_id}")
        reply = chat_once(session_id, chat_request.message)
        logger.debug(f"Generated reply for session {session_id}")
        return ChatResponse(reply=reply, session_id=session_id)
    except Exception as e:
        logger.exception(f"Error in chat endpoint for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=HistoryResponse)
async def history(session_id: str = Query(..., description="Session ID to get history for")):
    logger.info(f"Fetching history for session: {session_id}")
    try:
        hist = get_history(session_id)
        logger.debug(f"Found {len(hist)} messages in history for session {session_id}")
        return HistoryResponse(session_id=session_id, history=hist)
    except Exception as e:
        logger.exception(f"Error fetching history for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/history/reset")
async def reset_history(session_id: str = None):
    try:
        if session_id:
            logger.info(f"Resetting history for session: {session_id}")
            reset_history(session_id)
            return {"status": "success", "message": f"History cleared for session {session_id}"}
        else:
            logger.info("Resetting all chat histories")
            reset_history()
            return {"status": "success", "message": "All chat histories cleared"}
    except Exception as e:
        logger.exception("Error resetting history")
        raise HTTPException(status_code=500, detail=str(e))