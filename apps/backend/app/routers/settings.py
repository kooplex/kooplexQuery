from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.schemas.settings import ConfigPayload, ConnectRequest
from app.services import config_store
from app.services.kooplex_bridge import (
    create_chat_database_if_missing,
    get_db_chat,
    get_db_source,
    persist_db_config_env,
)
from app.services.runtime_state import runtime_state

router = APIRouter()


def _dump_model(payload: ConfigPayload | ConnectRequest, *, by_alias: bool = False) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(by_alias=by_alias)
    return payload.dict(by_alias=by_alias)


def _loaded_config_id() -> int | None:
    if not runtime_state.active_config:
        return None
    config_id = runtime_state.active_config.get("id")
    return config_id if isinstance(config_id, int) else None


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
        # Metadata sync is best-effort and should not block connectivity/config actions.
        return


@router.get("/database-configs")
def list_database_configs() -> dict[str, object]:
    config_store.init_store()
    return {"items": config_store.list_configs()}


@router.post("/database-configs")
def save_database_config(payload: ConfigPayload) -> dict[str, object]:
    config_store.init_store()
    record_id = config_store.save_config(_dump_model(payload, by_alias=True))
    return {"id": record_id}


@router.put("/database-configs/{config_id}")
def update_database_config(config_id: int, payload: ConfigPayload) -> dict[str, object]:
    config_store.init_store()
    try:
        updated = config_store.update_config(config_id, _dump_model(payload, by_alias=True))
    except config_store.DuplicateConfigTitleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if not updated:
        raise HTTPException(status_code=404, detail="Configuration not found")

    if runtime_state.active_config and runtime_state.active_config.get("id") == config_id:
        row = config_store.get_config(config_id)
        if row is not None:
            runtime_cfg = config_store.to_runtime_config(row)
            runtime_cfg["id"] = config_id
            runtime_state.active_config = runtime_cfg
            persist_db_config_env(runtime_cfg)
            _sync_sqlite_metadata_from_chat_db()

    return {"updated": True, "id": config_id}


@router.delete("/database-configs/{config_id}")
def delete_database_config(config_id: int) -> dict[str, object]:
    config_store.init_store()
    deleted = config_store.delete_config(config_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Configuration not found")
    if runtime_state.active_config and runtime_state.active_config.get("id") == config_id:
        runtime_state.active_config = None
        config_store.set_active_config_id(None)
    return {"deleted": True}


@router.post("/database-configs/select/{config_id}")
def select_database_config(config_id: int) -> dict[str, object]:
    config_store.init_store()
    row = config_store.get_config(config_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Configuration not found")
    runtime_cfg = config_store.to_runtime_config(row)
    runtime_cfg["id"] = config_id
    runtime_state.active_config = runtime_cfg
    config_store.set_active_config_id(config_id)
    persist_db_config_env(runtime_cfg)
    _sync_sqlite_metadata_from_chat_db()
    return {"active": runtime_cfg}


@router.get("/database-configs/active")
def get_active_config() -> dict[str, object]:
    return {"active": runtime_state.active_config}


@router.post("/connect")
def connect_databases(payload: ConnectRequest) -> dict[str, object]:
    data = _dump_model(payload, by_alias=True)
    db_cfg = data["database_server"]
    chat_cfg = data["chat_database_server"]

    if payload.create_chat_database_if_missing:
        create_chat_database_if_missing(
            hostname=chat_cfg.get("hostname"),
            port=int(chat_cfg.get("port") or 5432),
            database=chat_cfg.get("database"),
        )

    runtime_state.active_config = data
    persist_db_config_env(data)
    db_source = get_db_source()

    # Connectivity check to source DB.
    db_source.query("SELECT 1")

    # Connectivity check to chat DB and schema bootstrap.
    db_chat = get_db_chat()
    with db_chat.engine.connect() as con:
        con.execute(text("SELECT 1"))

    if payload.save:
        config_store.init_store()
        runtime_state.active_config["id"] = config_store.save_config(data)
        config_store.set_active_config_id(runtime_state.active_config["id"])
        _sync_sqlite_metadata_from_chat_db()

    return {"connected": True, "active": runtime_state.active_config}
