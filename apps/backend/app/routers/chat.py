from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.schemas.chat import (
    CreateSessionRequest,
    RunQueryRequest,
    SaveChatItemRequest,
    SaveQueryRequest,
    StreamChatRequest,
)
from app.services.kooplex_bridge import fetch_chat_history, get_db_chat, get_db_source, stream_chat_chunks
from app.services.kooplex_bridge import (
    add_model as add_model_entry,
    delete_model as delete_model_entry,
    list_models as list_model_entries,
)

router = APIRouter()


class AddModelRequest(BaseModel):
    model_name: str
    provider: str | None = None


@router.post("/sessions")
def create_session(payload: CreateSessionRequest) -> dict[str, object]:
    db_chat = get_db_chat()
    session_id = db_chat.new_session(
        username=payload.username,
        email=payload.email,
        label=payload.label,
        meta=payload.meta,
        referenced_session=payload.referenced_session_id,
    )
    return {"session_id": session_id}


@router.get("/sessions/{session_id}/history")
def get_session_history(session_id: int) -> dict[str, object]:
    return {"items": fetch_chat_history(session_id)}


@router.post("/messages")
def save_message_pair(payload: SaveChatItemRequest) -> dict[str, object]:
    db_chat = get_db_chat()
    db_chat.save_chat_item(
        session_id=payload.session_id,
        user_prompt=payload.user_prompt,
        agent_response=payload.agent_response,
        model_name=payload.model_name,
    )
    return {"saved": True}


@router.post("/queries")
def save_query(payload: SaveQueryRequest) -> dict[str, object]:
    db_chat = get_db_chat()
    db_chat.save_query(
        session_id=payload.session_id,
        question_content=payload.question_content,
        sql=payload.sql,
        question_type=payload.question_type,
        public=payload.public,
    )
    return {"saved": True}


@router.post("/query")
def run_query(payload: RunQueryRequest) -> dict[str, object]:
    db_source = get_db_source()
    sql = payload.sql.strip().rstrip(";")
    result = db_source.query(sql)
    rows = result.fetchall()
    if hasattr(result, "keys"):
        columns = list(result.keys())
    elif rows and hasattr(rows[0], "_mapping"):
        columns = list(rows[0]._mapping.keys())
    elif rows and isinstance(rows[0], (list, tuple)):
        columns = [f"col_{idx + 1}" for idx in range(len(rows[0]))]
    else:
        columns = []
    return {
        "columns": columns,
        "rows": [list(row) for row in rows],
        "count": len(rows),
    }


@router.post("/stream")
async def stream_chat_response(payload: StreamChatRequest) -> StreamingResponse:
    async def generate():
        collected: list[str] = []
        try:
            async for chunk in stream_chat_chunks(payload.prompt, payload.model_name):
                collected.append(chunk)
                yield chunk
        except Exception as exc:
            yield f"\n[stream-error] {exc}"
        finally:
            if payload.session_id is not None and collected:
                try:
                    db_chat = get_db_chat()
                    db_chat.save_chat_item(
                        session_id=payload.session_id,
                        user_prompt=payload.prompt,
                        agent_response="".join(collected),
                        model_name=payload.model_name,
                    )
                except Exception:
                    # Keep streaming path resilient; history can still be saved manually from UI.
                    pass

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@router.get("/models")
def list_models() -> dict[str, object]:
    return {"items": list_model_entries()}


@router.post("/models")
def add_model(payload: AddModelRequest) -> dict[str, object]:
    return {"item": add_model_entry(payload.model_name, payload.provider)}


@router.delete("/models")
def delete_model(model_name: str, provider: str | None = None) -> dict[str, object]:
    return {"deleted": delete_model_entry(model_name, provider)}
