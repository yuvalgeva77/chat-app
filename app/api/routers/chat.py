from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
from uuid import uuid4
import json

from app.api.schemas.chat import ChatRequest, ChatResponse, HistoryResponse
from app.api.services.chat_engine import chat_once, chat_stream, get_history, reset_history
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger("chat-router")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check api called")
    return {"status": "ok", "message": "Chat service is running"}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """
    Handle chat messages and return AI responses.

    - If chat_request.session_id is missing, a new UUID is created.
    - Returns the assistant's reply and the session_id to reuse on the client.
    """
    logger.debug("Chat endpoint called")
    try:
        # Use provided session ID or generate a new one
        session_id = chat_request.session_id or str(uuid4())

        # Log the request
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Chat request from {client_ip} | Session: {session_id}")

        # Get response from chat engine
        reply = chat_once(session_id, chat_request.message)
        logger.debug(f"Chat response: {reply[:400]}")

        return ChatResponse(reply=reply, session_id=session_id)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/stream")
async def chat_stream_endpoint(
    request: Request,
    session_id: Optional[str] = Query(None, description="Existing session id; if omitted a new one is created"),
    message: str = Query(..., description="User message to stream a response for"),
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    The client receives tokens as they're generated.

    Event payloads:
      data: {"token": "<chunk>"}  (multiple)
      data: {"done": true}        (exactly once, at end)
    """
    sid = session_id or str(uuid4())
    logger.info(f"SSE stream start | Session: {sid}")

    def event_generator():
        try:
            for chunk in chat_stream(sid, message):
                yield f"data: {json.dumps({'token': chunk, 'session_id': sid})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': sid})}\n\n"
        except Exception as e:
            logger.error(f"SSE error: {e}")
            # send an error event (optional)
            yield f"event: error\ndata: {json.dumps({'error': str(e), 'session_id': sid})}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

@router.get("/history", response_model=HistoryResponse)
async def get_chat_history(
    session_id: str = Query(..., description="Session ID to get history for")
):
    """Get conversation history for a specific session."""
    try:
        logger.debug(f"Get history for session {session_id}")
        history = get_history(session_id)
        return HistoryResponse(session_id=session_id, history=history)
    except Exception as e:
        logger.error(f"Error getting history for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/history/reset")
async def reset_chat_history(
    session_id: Optional[str] = Query(
        None,
        description="Optional session ID to reset. If not provided, resets all sessions."
    )
):
    """
    Reset conversation history for a specific session or all sessions.
    """
    try:
        logger.debug(f"Reset history for session {session_id}")
        count = reset_history(session_id)
        if session_id:
            message = f"Reset history for session {session_id}"
        else:
            message = "Reset all chat histories"

        logger.info(f"{message} | Sessions affected: {count}")
        return {"status": "success", "message": message, "sessions_reset": count}
    except Exception as e:
        logger.error(f"Error resetting history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
