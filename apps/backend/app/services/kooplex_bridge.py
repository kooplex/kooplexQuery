from __future__ import annotations

import importlib.util
import logging
import os
import sqlite3
import sys
from typing import AsyncIterator
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import text

from app.services.runtime_state import runtime_state


REPO_ROOT = Path(__file__).resolve().parents[4]
KOOPLEX_SRC = REPO_ROOT / "kooplexQuery"
logger = logging.getLogger(__name__)


def _load_module(module_name: str, file_path: Path):
    cached = sys.modules.get(module_name)
    if cached is not None:
        # Reuse only if this cache entry belongs to the same file and is usable.
        if getattr(cached, "__file__", None) == str(file_path):
            return cached
        sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Prevent stale partially initialized modules from poisoning future imports.
        sys.modules.pop(module_name, None)
        raise
    return module


def get_active_config() -> dict:
    if runtime_state.active_config is None:
        raise HTTPException(status_code=400, detail="No active database configuration. Connect first.")
    return runtime_state.active_config


def get_db_chat():
    db_chat_mod = _load_module("kooplex_db_chat_module", KOOPLEX_SRC / "db_chat.py")
    DBChat = db_chat_mod.DBChat

    cfg = get_active_config()["chat_database_server"]
    return DBChat(
        hostname=cfg.get("hostname"),
        port=int(cfg.get("port") or 5432),
        database=cfg.get("database"),
        schema=cfg.get("schema") or "public",
        user=cfg.get("user"),
        password=cfg.get("password"),
    )


def create_chat_database_if_missing(hostname: str, port: int, database: str) -> bool:
    db_chat_mod = _load_module("kooplex_db_chat_module", KOOPLEX_SRC / "db_chat.py")
    DBChat = db_chat_mod.DBChat
    return DBChat.create_database_if_missing(hostname=hostname, port=port, database=database)


def get_db_source():
    db_mod = _load_module("kooplex_db_module", KOOPLEX_SRC / "db.py")
    DBQuery = db_mod.DBQuery

    cfg = get_active_config()["database_server"]
    return DBQuery(
        hostname=cfg.get("hostname"),
        port=int(cfg.get("port") or 5432),
        database=cfg.get("database"),
        schema=cfg.get("schema") or "public",
        user=cfg.get("user"),
        password=cfg.get("password"),
        url=cfg.get("url"),
        type=cfg.get("type") or "postgresql",
        title=cfg.get("title"),
    )


def get_vectorstore():
    module_name = "kooplex_vectorstore_module"
    module_path = KOOPLEX_SRC / "utils" / "vectorstore.py"
    vectorstore_mod = _load_module(module_name, module_path)
    VectorStore = getattr(vectorstore_mod, "VectorStore", None)
    if VectorStore is None:
        # Retry once from a clean module cache in case a previous import left stale state.
        sys.modules.pop(module_name, None)
        vectorstore_mod = _load_module(module_name, module_path)
        VectorStore = getattr(vectorstore_mod, "VectorStore", None)
    if VectorStore is None:
        exported = ", ".join(sorted(name for name in dir(vectorstore_mod) if not name.startswith("_")))
        raise RuntimeError(f"VectorStore class not found in {module_path}. Exported names: {exported}")

    cfg = get_active_config()["database_server"]
    title = cfg.get("title") or "kooplexquery_db"
    safe_title = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(title)).strip("_") or "kooplexquery_db"
    persist_dir = Path(__file__).resolve().parents[2] / "data" / "vectorstore" / safe_title
    return VectorStore(persist_directory=str(persist_dir))


def model_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "llm_models.sqlite3"


def _ensure_model_registry() -> Path:
    db_path = model_registry_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4.1-mini")
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                UNIQUE(model_name, provider)
            )
            """
        )
        con.execute(
            "INSERT OR IGNORE INTO llm_models (model_name, provider) VALUES (?, ?)",
            (default_model, default_provider),
        )
        con.commit()
    return db_path


def list_models() -> list[dict]:
    db_path = _ensure_model_registry()
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT provider, model_name FROM llm_models ORDER BY provider, model_name"
        ).fetchall()
    return [{"provider": row[0], "model_name": row[1]} for row in rows]


def add_model(model_name: str, provider: str | None = None) -> dict:
    if not model_name or not model_name.strip():
        raise HTTPException(status_code=400, detail="model_name cannot be empty")
    normalized_name = model_name.strip()
    normalized_provider = (provider or "openai").strip()
    db_path = _ensure_model_registry()
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT OR IGNORE INTO llm_models (model_name, provider) VALUES (?, ?)",
            (normalized_name, normalized_provider),
        )
        con.commit()
    return {"provider": normalized_provider, "model_name": normalized_name}


def delete_model(model_name: str, provider: str | None = None) -> bool:
    if not model_name or not model_name.strip():
        raise HTTPException(status_code=400, detail="model_name cannot be empty")
    db_path = _ensure_model_registry()
    with sqlite3.connect(db_path) as con:
        if provider:
            cur = con.execute(
                "DELETE FROM llm_models WHERE model_name = ? AND provider = ?",
                (model_name.strip(), provider.strip()),
            )
        else:
            cur = con.execute(
                "DELETE FROM llm_models WHERE model_name = ?",
                (model_name.strip(),),
            )
        con.commit()
        deleted = cur.rowcount > 0
    _ensure_model_registry()
    return deleted


def fetch_chat_history(session_id: int) -> list[dict]:
    db_chat = get_db_chat()
    schema_name = db_chat._quote_ident(db_chat.schema)
    q = text(
        f"""
        SELECT id, session_id, sequence, role, timestamp, content
        FROM {schema_name}.chathistory
        WHERE session_id = :session_id
        ORDER BY sequence, id
        """
    )
    with db_chat.engine.connect() as con:
        rows = con.execute(q, {"session_id": session_id}).mappings().all()
    items = [dict(row) for row in rows]

    # The original Motor seeds each session with a system instruction/context
    # in memory, but that message is not persisted into chathistory.
    # Surface a synthetic system row so the UI can show what was preloaded.
    instruction_parts: list[str] = []
    for reference in ("data_descriptor", "reference", "schema", "instruction"):
        try:
            content = db_chat.load_knowledge(reference=reference)
        except Exception:
            content = None
        text_value = (content or "").strip()
        if not text_value:
            continue
        if reference == "instruction":
            instruction_parts.append(text_value)
        else:
            instruction_parts.append(f"[{reference}]\n{text_value}")

    if instruction_parts:
        synthetic = {
            "id": 0,
            "session_id": session_id,
            "sequence": 0,
            "role": "system",
            "timestamp": "",
            "content": "\n\n".join(instruction_parts),
        }
        return [synthetic, *items]

    return items


def _build_initial_system_message(db_chat, db_source, table_name_filter: str = "%") -> str:
    default_instruction = (
        "You are a helpful assistant that translates natural language questions into SQL queries. "
        "Always try to answer the question with a SQL query based on the provided database schema "
        "and data description. If the question is not clear, ask for clarification. Always use the "
        "provided schema and data description to inform your SQL generation. Do not make up any "
        "information about the database that is not included in the schema or data description."
    )

    context = ""
    try:
        data_descriptor = db_chat.load_knowledge(reference="data_descriptor")
        dbschema = db_chat.load_knowledge(reference="schema")
        dbreference = db_chat.load_knowledge(reference="reference")
        table_description = dict(db_source.describe_tables(table_name_filter))
        table_column_description = str(db_source.describe_columns(table_name_filter))
        context = (
            f"{data_descriptor}\n\n"
            f"{dbreference} table descriptions: {table_description}\n\n"
            f"{dbreference} table column descriptions: {table_column_description}\n\n"
            f"Database Schema: {dbschema}"
        )
    except Exception:
        context = ""

    try:
        instruction = db_chat.load_knowledge(reference="instruction")
    except Exception:
        instruction = default_instruction

    system_parts: list[str] = []
    if context:
        system_parts.append(context)
    if instruction:
        system_parts.append(str(instruction))
    if not system_parts:
        system_parts.append(default_instruction)
    return "\n\n".join(system_parts)


async def stream_chat_chunks(prompt: str, model_name: str = "api") -> AsyncIterator[str]:
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt cannot be empty")

    motor_mod = _load_module("kooplex_motor_module", KOOPLEX_SRC / "motor.py")
    history_mod = _load_module("kooplex_history_module", KOOPLEX_SRC / "history.py")

    Motor = motor_mod.Motor
    CustomChatHistory = history_mod.CustomChatHistory

    cfg = get_active_config()
    db_cfg = cfg["database_server"]
    chat_cfg = cfg["chat_database_server"]

    # Build a lightweight Motor instance bound to the active DB configs.
    # This keeps streaming stateless while reusing the existing model/provider logic.
    motor = object.__new__(Motor)
    motor._table_name_filter = "%"
    motor.db_source = motor._dbtarget_init(db_cfg)
    motor.db_chat = motor._dbchat_init(chat_cfg)
    motor._ensure_sync_manager()
    motor.current_model = model_name or "api"
    motor._chat_history = CustomChatHistory()

    initial_system_message = _build_initial_system_message(
        db_chat=motor.db_chat,
        db_source=motor.db_source,
        table_name_filter=motor._table_name_filter,
    )
    if initial_system_message:
        motor._chat_history.add_system_message(initial_system_message)

    motor._chat_history.add_user_message(prompt, metadata={"model": motor.current_model})

    llm_messages_payload = [
        {
            "role": getattr(message, "type", "unknown"),
            "content": str(getattr(message, "content", "")),
        }
        for message in motor._chat_history.messages
    ]
    logger.info("Submitting LLM stream payload: %s", llm_messages_payload)

    async for chunk in motor._llm_agent.astream(motor._chat_history.messages):
        content = getattr(chunk, "content", None)
        if content:
            yield str(content)
