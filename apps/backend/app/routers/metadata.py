from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.services import config_store
from app.services.kooplex_bridge import get_db_chat, get_db_source
from app.services.runtime_state import runtime_state

router = APIRouter()


class KnowledgeUpsertRequest(BaseModel):
    reference: str
    content: str


def _loaded_config_id() -> int | None:
    if runtime_state.active_config:
        config_id = runtime_state.active_config.get("id")
        if isinstance(config_id, int):
            return config_id
    return config_store.get_active_config_id()


def _sync_sqlite_metadata_from_chat_db() -> None:
    try:
        config_id = _loaded_config_id()
        if config_id is None:
            return
        db_chat = get_db_chat()
        keys, data = db_chat.fetch_all_knowledge()
        items = [dict(zip(keys, row)) for row in data]
        config_store.init_store()
        config_store.sync_metadata_from_knowledge(items, config_id)
    except Exception:
        # Do not block metadata CRUD if local sync is unavailable.
        return


@router.get("/knowledge")
def list_knowledge() -> dict[str, object]:
    db_chat = get_db_chat()
    keys, data = db_chat.fetch_all_knowledge()
    items = [dict(zip(keys, row)) for row in data]
    return {"items": items}


@router.post("/knowledge")
def create_knowledge(payload: KnowledgeUpsertRequest) -> dict[str, object]:
    db_chat = get_db_chat()
    db_chat.save_knowledge(payload.reference, payload.content)
    _sync_sqlite_metadata_from_chat_db()
    return {"saved": True, "reference": payload.reference}


@router.put("/knowledge/{reference}")
def update_knowledge(reference: str, payload: KnowledgeUpsertRequest) -> dict[str, object]:
    db_chat = get_db_chat()
    db_chat.save_knowledge(reference, payload.content)
    _sync_sqlite_metadata_from_chat_db()
    return {"updated": True, "reference": reference}


@router.delete("/knowledge/{knowledge_id}")
def delete_knowledge(knowledge_id: int) -> dict[str, object]:
    db_chat = get_db_chat()
    schema_name = db_chat._quote_ident(db_chat.schema)
    q = text(f"DELETE FROM {schema_name}.knowledge WHERE id = :id")
    with db_chat.engine.begin() as con:
        con.execute(q, {"id": knowledge_id})
    _sync_sqlite_metadata_from_chat_db()
    return {"deleted": True, "knowledge_id": knowledge_id}


@router.get("/tables")
def describe_tables() -> dict[str, object]:
    db_source = get_db_source()
    rows = db_source.describe_tables()
    items = [{"table": row[0], "description": row[1]} for row in rows]
    return {"items": items}


@router.get("/columns")
def describe_columns() -> dict[str, object]:
    db_source = get_db_source()
    rows = db_source.describe_columns()
    items = [
        {
            "table": row[0],
            "column": row[1],
            "description": row[2],
            "type": row[3],
        }
        for row in rows
    ]
    return {"items": items}
