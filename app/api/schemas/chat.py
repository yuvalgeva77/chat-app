from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str | None = Field(None, description="Conversation id. If None, a new one is created.")

class ChatResponse(BaseModel):
    reply: str
    session_id: str

class HistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]
