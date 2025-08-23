from fastapi import APIRouter, HTTPException, Request, Query
from typing import Optional
from uuid import uuid4
from app.api.schemas.chat import ChatRequest, ChatResponse, HistoryResponse
from app.api.services.chat_engine import chat_once, get_history, reset_history
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger("chat-router")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Chat service is running"}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """
    Handle chat messages and return AI responses.
    
    Args:
        request: The HTTP request object
        chat_request: The chat request containing the user's message and optional session ID
        
    Returns:
        ChatResponse containing the AI's reply and session ID
    """
    try:
        # Use provided session ID or generate a new one
        session_id = chat_request.session_id or str(uuid4())
        
        # Log the request
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Chat request from {client_ip} | Session: {session_id}")
        
        # Get response from chat engine
        reply = chat_once(session_id, chat_request.message)
        
        return ChatResponse(
            reply=reply,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=HistoryResponse)
async def get_chat_history(
    session_id: str = Query(..., description="Session ID to get history for")
):
    """
    Get conversation history for a specific session.
    
    Args:
        session_id: The session ID to retrieve history for
        
    Returns:
        HistoryResponse containing the conversation history
    """
    try:
        history = get_history(session_id)
        return HistoryResponse(
            session_id=session_id,
            history=history
        )
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
    
    Args:
        session_id: Optional session ID to reset. If None, resets all sessions.
        
    Returns:
        dict: Status message and count of sessions reset
    """
    try:
        count = reset_history(session_id)
        if session_id:
            message = f"Reset history for session {session_id}"
        else:
            message = f"Reset all chat histories"
            
        logger.info(f"{message} | Sessions affected: {count}")
        
        return {
            "status": "success",
            "message": message,
            "sessions_reset": count
        }
    except Exception as e:
        logger.error(f"Error resetting history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))