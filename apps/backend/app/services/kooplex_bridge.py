from __future__ import annotations

import importlib.util
import asyncio
import logging
import os
import re
import socket
import sqlite3
import sys
from typing import AsyncIterator
from pathlib import Path
from urllib.parse import urlparse
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

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


def persist_db_config_env(config: dict | None = None) -> Path:
    cfg = config if config is not None else get_active_config()
    db_cfg = cfg.get("database_server", {})

    env_path = REPO_ROOT / "config.env"
    lines = [
        f"DB_HOST={db_cfg.get('hostname') or 'localhost'}",
        f"DB_PORT={int(db_cfg.get('port') or 5432)}",
        f"DB_DATABASE={db_cfg.get('database') or 'public'}",
        f"DB_SCHEMA={db_cfg.get('schema') or 'public'}",
        f"DB_USER={db_cfg.get('user') or 'reader'}",
        f"DB_PASSWORD={db_cfg.get('password') or ''}",
        f"DB_TYPE={db_cfg.get('type') or 'postgresql'}",
        f"DB_URL={db_cfg.get('url') or ''}",
    ]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


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


def _parse_host_port(endpoint: str) -> tuple[str | None, int | None]:
    value = (endpoint or "").strip()
    if not value:
        return None, None

    # Handle full URLs first.
    if "://" in value:
        parsed = urlparse(value)
        if parsed.hostname:
            if parsed.port:
                return parsed.hostname, parsed.port
            return parsed.hostname, 443 if parsed.scheme == "https" else 80

    # Handle host:port and host:port/path formats.
    host_port = value.split("/", 1)[0]
    if ":" in host_port:
        host, port_text = host_port.rsplit(":", 1)
        if port_text.isdigit():
            return host.strip() or None, int(port_text)
    return host_port.strip() or None, None


def _provider_endpoint(provider: str | None) -> tuple[str | None, int | None]:
    normalized = (provider or "").strip().lower()
    if normalized == "openai":
        host, port = _parse_host_port(
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        return host, port or 443
    if normalized == "anthropic":
        host, port = _parse_host_port(
            os.getenv("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        )
        return host, port or 443

    # Local/custom providers are treated as host:port endpoints.
    return _parse_host_port(provider or "")


def _is_provider_reachable(provider: str | None, timeout_seconds: float = 1.5) -> bool:
    normalized = (provider or "").strip().lower()
    if normalized in {"openai", "anthropic"}:
        endpoint = (
            (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1")
            if normalized == "openai"
            else (os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com")
        )
        endpoint = endpoint.strip()
        if not endpoint:
            return False
        if "://" not in endpoint:
            endpoint = f"https://{endpoint}"

        # For cloud APIs, any HTTP response (including 401/403) means host is reachable.
        for method in ("HEAD", "GET"):
            try:
                req = urlrequest.Request(endpoint, method=method)
                with urlrequest.urlopen(req, timeout=max(timeout_seconds, 3.0)):
                    return True
            except HTTPError:
                return True
            except URLError:
                continue
            except Exception:
                continue
        return False

    host, port = _provider_endpoint(provider)
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=max(timeout_seconds, 3.0)):
            return True
    except OSError:
        return False


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

    provider_cache: dict[str | None, bool] = {}
    items: list[dict] = []
    for provider, model_name in rows:
        if provider not in provider_cache:
            provider_cache[provider] = _is_provider_reachable(provider)
        items.append(
            {
                "provider": provider,
                "model_name": model_name,
                "reachable": provider_cache[provider],
            }
        )
    return items


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


def fetch_sessions() -> list[dict]:
    db_chat = get_db_chat()
    schema_name = db_chat._quote_ident(db_chat.schema)
    q = text(
        f"""
        SELECT DISTINCT ON (session_id) session_id, timestamp, content
        FROM {schema_name}.chathistory
        WHERE role::text IN ('user', 'human')
        ORDER BY session_id DESC, sequence ASC, id ASC
        """
    )
    with db_chat.engine.connect() as con:
        rows = con.execute(q).mappings().all()
    return [dict(row) for row in rows]


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


def _plotting_runtime_instruction() -> str:
    return (
        "When you generate Python plotting code for this app, follow this runtime contract strictly:\n"
        "- Use Python code that loads DB settings from a config.env file via dotenv and creates SQLAlchemy engine.\n"
        "- The absolute path to config.env is available as: os.environ.get('KOOPLEX_CONFIG_ENV_PATH', 'config.env')\n"
        "- Use this pattern (replacing the sql query and plotting style) and adapt the connectionstring to DB_TYPE value:\n"
        "  from sqlalchemy import create_engine, text\n"
        "  import os\n"
        "  import pandas as pd\n"
        "  from dotenv import load_dotenv\n"
        "  load_dotenv(os.environ.get('KOOPLEX_CONFIG_ENV_PATH', 'config.env'))\n"
        "  _ge = lambda x, d: os.getenv(x, d)\n"
        "  hostname = _ge('DB_HOST', 'localhost')\n"
        "  port = int(_ge('DB_PORT', 5432))\n"
        "  database = _ge('DB_DATABASE', 'public')\n"
        "  schema = _ge('DB_SCHEMA', 'public')\n"
        "  db_user = _ge('DB_USER', 'reader')\n"
        "  db_password = _ge('DB_PASSWORD', '')\n"
        "  db_type = _ge('DB_TYPE', 'postgresql')\n"
        "  if db_type == 'mssql':\n"
        "    connectionstring = f'mssql+pymssql://{db_user}:{db_password}@{hostname}:{port}/{database}'\n"
        "  else:\n"
        "    connectionstring = f'postgresql+psycopg2://{db_user}:{db_password}@{hostname}:{port}/{database}'\n"
        "  engine = create_engine(connectionstring)\n"
        "  df = pd.read_sql(text(SQL_QUERY), engine, params={})\n"
        "- Replace SQL_QUERY with the actual SQL that answers the user's question.\n"
        "- Build a Plotly figure and assign it to variable `fig`.\n"
        "- Return code in a single Python code block without explanation around it."
    )


def _build_plotting_system_message(db_chat) -> str:
    parts: list[str] = [_plotting_runtime_instruction()]

    # Optional user-provided plotting instructions stored in knowledge table.
    for ref in ("plotting_instruction", "plotting_instructions"):
        try:
            extra = (db_chat.load_knowledge(reference=ref) or "").strip()
        except Exception:
            extra = ""
        if extra:
            parts.append(f"[user_plotting_instruction]\n{extra}")

    return "\n\n".join(parts)


def _get_latest_saved_sql_for_session(db_chat, session_id: int) -> str:
    schema_name = db_chat._quote_ident(db_chat.schema)
    q = text(
        f"""
        SELECT qy.sql
        FROM {schema_name}.query qy
        JOIN {schema_name}.question qs ON qs.id = qy.question_id
        WHERE qs.session_id = :session_id
        ORDER BY qy.id DESC
        LIMIT 1
        """
    )
    with db_chat.engine.connect() as con:
        value = con.execute(q, {"session_id": session_id}).scalar()
    return (value or "").strip()


def _build_initial_system_message(db_chat, db_source, table_name_filter: str = "%") -> str:
    default_instruction = (
        "You are a helpful assistant that translates natural language questions into SQL queries. "
        "Always try to answer the question with a SQL query based on the provided database schema "
        "and data description. If the question is not clear, ask for clarification. Always use the "
        "provided schema and data description to inform your SQL generation. Do not make up any "
        "information about the database that is not included in the schema or data description."
    )

    plotting_runtime_instruction = _plotting_runtime_instruction()

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
    # system_parts.append(plotting_runtime_instruction)
    return "\n\n".join(system_parts)


async def generate_plot_code(
    session_id: int,
    model_name: str = "api",
    model_provider: str | None = None,
    sql_override: str | None = None,
    plotting_request: str | None = None,
) -> dict[str, str]:
    if session_id <= 0:
        raise HTTPException(status_code=400, detail="session_id must be a positive integer")

    sql_text = (sql_override or "").strip()

    cfg = get_active_config()
    db_cfg = cfg["database_server"]
    chat_cfg = cfg["chat_database_server"]

    motor_mod = _load_module("kooplex_motor_module", KOOPLEX_SRC / "motor.py")
    Motor = motor_mod.Motor

    motor = object.__new__(Motor)
    motor._table_name_filter = "%"
    motor.db_source = motor._dbtarget_init(db_cfg)
    motor.db_chat = motor._dbchat_init(chat_cfg)
    motor._ensure_sync_manager()

    if not sql_text:
        sql_text = _get_latest_saved_sql_for_session(motor.db_chat, session_id)
    if not sql_text:
        raise HTTPException(status_code=400, detail="No SQL found for this session. Save or provide SQL first.")

    selected_model_name = (model_name or "api").strip() or "api"
    selected_provider = model_provider.strip() if model_provider else None
    if selected_provider:
        motor.current_model = motor_mod.LLM_Model(
            provider=selected_provider,
            model_name=selected_model_name,
        )
    else:
        motor.current_model = selected_model_name

    system_msg = _build_plotting_system_message(motor.db_chat)
    extra_request = (plotting_request or "").strip()
    prompt = (
        f"{system_msg}\n\n"
        "Task: Generate Python Plotly code for the following SQL result.\n"
        f"SQL:\n```sql\n{sql_text}\n```\n"
        f"Additional plotting request: {extra_request if extra_request else 'None'}\n"
        "Output only one Python code block."
    )

    timeout_seconds = int(os.getenv("PLOT_CODE_TIMEOUT_SECONDS", "45"))
    try:
        resp = await asyncio.wait_for(motor._llm_agent.ainvoke(prompt), timeout=timeout_seconds)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Plot code generation failed: {exc}") from exc

    generated = (getattr(resp, "content", "") or "").strip()
    if not generated:
        raise HTTPException(status_code=500, detail="Empty plot code response from model")

    if "```" not in generated:
        generated = f"```python\n{generated}\n```"

    return {"code": generated, "sql_used": sql_text}


def _extract_sql_from_model_output(content: str) -> str:
    text_value = (content or "").strip()
    if not text_value:
        return ""

    sql_block = re.search(r"```sql\s*([\s\S]*?)```", text_value, flags=re.IGNORECASE)
    if sql_block:
        return sql_block.group(1).strip()

    any_block = re.search(r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)```", text_value)
    if any_block:
        return any_block.group(1).strip()

    return text_value


async def correct_sql_from_error(
    sql: str,
    error_message: str,
    history_summary: str | None = None,
    model_name: str = "api",
    model_provider: str | None = None,
) -> dict[str, str]:
    sql_text = (sql or "").strip()
    if not sql_text:
        raise HTTPException(status_code=400, detail="sql cannot be empty")

    error_text = (error_message or "").strip()
    if not error_text:
        raise HTTPException(status_code=400, detail="error_message cannot be empty")

    history_text = (history_summary or "").strip()

    motor_mod = _load_module("kooplex_motor_module", KOOPLEX_SRC / "motor.py")
    Motor = motor_mod.Motor

    cfg = get_active_config()
    db_cfg = cfg["database_server"]
    chat_cfg = cfg["chat_database_server"]

    motor = object.__new__(Motor)
    motor._table_name_filter = "%"
    motor.db_source = motor._dbtarget_init(db_cfg)
    motor.db_chat = motor._dbchat_init(chat_cfg)
    motor._ensure_sync_manager()

    selected_model_name = (model_name or "api").strip() or "api"
    selected_provider = model_provider.strip() if model_provider else None
    if selected_provider:
        motor.current_model = motor_mod.LLM_Model(
            provider=selected_provider,
            model_name=selected_model_name,
        )
    else:
        motor.current_model = selected_model_name

    try:
        table_description = dict(motor.db_source.describe_tables(motor._table_name_filter))
        table_column_description = str(motor.db_source.describe_columns(motor._table_name_filter))
        schema_context = (
            f"Table descriptions: {table_description}\n\n"
            f"Table column descriptions: {table_column_description}"
        )
    except Exception:
        schema_context = ""

    prompt = (
        "You are an expert SQL fixer. Correct the SQL query based on the database schema and error message.\n"
        "Rules:\n"
        "- Return only corrected SQL in a single ```sql code block.\n"
        "- Keep original intent and selected columns unless the error forces a change.\n"
        "- Use only tables/columns present in schema context.\n"
        "- Do not include any explanation outside the SQL code block.\n\n"
        f"Schema context:\n{schema_context}\n\n"
        f"Recent chat summary:\n{history_text if history_text else '(none)'}\n\n"
        f"Current SQL:\n```sql\n{sql_text}\n```\n\n"
        f"Error message:\n{error_text}\n"
    )

    timeout_seconds = int(os.getenv("CORRECT_SQL_TIMEOUT_SECONDS", "45"))
    try:
        resp = await asyncio.wait_for(motor._llm_agent.ainvoke(prompt), timeout=timeout_seconds)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SQL correction failed: {exc}") from exc

    raw_response = (getattr(resp, "content", "") or "").strip()
    corrected_sql = _extract_sql_from_model_output(raw_response)
    if not corrected_sql:
        raise HTTPException(status_code=500, detail="Empty SQL correction from model")

    return {
        "corrected_sql": corrected_sql,
        "raw_response": raw_response,
    }


async def stream_chat_chunks(
    prompt: str,
    model_name: str = "api",
    model_provider: str | None = None,
) -> AsyncIterator[str]:
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
    selected_model_name = (model_name or "api").strip() or "api"
    selected_provider = model_provider.strip() if model_provider else None
    if selected_provider:
        motor.current_model = motor_mod.LLM_Model(
            provider=selected_provider,
            model_name=selected_model_name,
        )
    else:
        motor.current_model = selected_model_name
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


async def prepare_question_for_validation(
    session_id: int,
    sql: str,
    model_name: str = "api",
    model_provider: str | None = None,
) -> str:
    if session_id <= 0:
        raise HTTPException(status_code=400, detail="session_id must be a positive integer")

    sql_text = (sql or "").strip()
    if not sql_text:
        raise HTTPException(status_code=400, detail="sql cannot be empty")

    history_rows = fetch_chat_history(session_id)
    history_lines: list[str] = []
    for row in history_rows:
        role = str(row.get("role") or "").lower()
        if role == "system":
            continue
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role in {"user", "human"} else "Assistant"
        history_lines.append(f"{label}: {content}")

    history_text = "\n".join(history_lines)
    prompt = (
        f"For last SQL {sql_text} based on the chat history\n{history_text}\n"
        "what was the question this sql query belongs to?\n"
        "If the final question is not in the chat history, please generate a new question "
        "else repeat the question from the chat history.\n"
        "Pay attention to user changes to SQL code and match it to the question.\n"
        "Just provide the question without any extra explanation."
    )

    motor_mod = _load_module("kooplex_motor_module", KOOPLEX_SRC / "motor.py")
    Motor = motor_mod.Motor

    cfg = get_active_config()
    db_cfg = cfg["database_server"]
    chat_cfg = cfg["chat_database_server"]

    motor = object.__new__(Motor)
    motor._table_name_filter = "%"
    motor.db_source = motor._dbtarget_init(db_cfg)
    motor.db_chat = motor._dbchat_init(chat_cfg)
    motor._ensure_sync_manager()

    selected_model_name = (model_name or "api").strip() or "api"
    selected_provider = model_provider.strip() if model_provider else None
    if selected_provider:
        motor.current_model = motor_mod.LLM_Model(
            provider=selected_provider,
            model_name=selected_model_name,
        )
    else:
        motor.current_model = selected_model_name

    timeout_seconds = int(os.getenv("SAVE_PREPARE_TIMEOUT_SECONDS", "45"))
    try:
        resp = await asyncio.wait_for(motor._llm_agent.ainvoke(prompt), timeout=timeout_seconds)
        generated = (getattr(resp, "content", "") or "").strip()
        if generated:
            return generated
    except Exception as exc:
        logger.warning("prepare_question_for_validation failed: %s", exc)

    for row in reversed(history_rows):
        role = str(row.get("role") or "").lower()
        if role in {"user", "human"}:
            content = str(row.get("content") or "").strip()
            if content:
                return content

    return f"What question is answered by this SQL query? {sql_text}".strip()
