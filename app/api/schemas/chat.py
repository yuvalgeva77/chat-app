from typing import Optional, List, Dict
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str

class HistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]
