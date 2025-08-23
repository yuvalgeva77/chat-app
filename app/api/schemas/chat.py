from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Message(BaseModel):
    """Represents a single message in the conversation."""
    role: str = Field(..., description="The role of the message sender ('user' or 'assistant')")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    message: str = Field(..., description="The user's message")
    session_id: Optional[str] = Field(
        None, 
        description="Optional session ID. If not provided, a new session will be created."
    )

class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    reply: str = Field(..., description="The assistant's reply")
    session_id: str = Field(..., description="The session ID for this conversation")

class HistoryResponse(BaseModel):
    """Response model for the history endpoint."""
    session_id: str = Field(..., description="The session ID")
    history: List[Message] = Field(..., description="List of messages in the conversation")
