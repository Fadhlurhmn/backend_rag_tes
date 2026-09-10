from pydantic import BaseModel
from typing import Optional
import uuid


class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    agent: str  # "manager" | "specialist"
    input_tokens: int
    output_tokens: int
    total_tokens: int
    conversation_id: str
    message_id: str


class HistoryMessage(BaseModel):
    id: str
    role: str
    content: str
    agent_role: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    created_at: str


class HistoryResponse(BaseModel):
    conversation_id: str
    messages: list[HistoryMessage]
