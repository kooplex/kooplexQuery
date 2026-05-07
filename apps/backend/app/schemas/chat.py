from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    username: str
    email: str
    label: Optional[str] = None
    meta: str = ""
    referenced_session_id: Optional[int] = None


class SaveChatItemRequest(BaseModel):
    session_id: int
    user_prompt: str
    agent_response: str
    model_name: str = "api"


class SaveQueryRequest(BaseModel):
    session_id: int
    question_content: str
    sql: str
    question_type: str = "user"
    public: bool = True


class RunQueryRequest(BaseModel):
    sql: str


class StreamChatRequest(BaseModel):
    prompt: str
    model_name: str = "api"
    session_id: Optional[int] = None
