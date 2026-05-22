from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "database_servers.sqlite"


class DuplicateConfigTitleError(ValueError):
    pass


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _ensure_database_servers_columns(con: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in con.execute("PRAGMA table_info(database_servers)").fetchall()
    }
    required = {
        "db_tag": "TEXT",
        "db_publication": "TEXT",
        "db_short_description": "TEXT",
    }
    for column, col_type in required.items():
        if column in columns:
            continue
        con.execute(f"ALTER TABLE database_servers ADD COLUMN {column} {col_type}")


def _normalize_lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _compact_lookup_key(value: str) -> str:
    return _normalize_lookup_key(value).replace(" ", "")


def _build_title_aliases(title: str) -> list[str]:
    normalized = _normalize_lookup_key(title)
    compact = _compact_lookup_key(title)
    stripped = _normalize_lookup_key(re.sub(r"\b(db|database)\b", "", title or "", flags=re.IGNORECASE))
    stripped_compact = stripped.replace(" ", "")

    aliases: list[str] = []
    for item in [normalized, compact, stripped, stripped_compact]:
        if item and item not in aliases:
            aliases.append(item)
    return aliases


def _first_present(record: dict, keys: list[str]):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _try_parse_structured_object(value: str | None) -> dict | None:
    text = (value or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    try:
        import yaml  # type: ignore

        parsed_yaml = yaml.safe_load(text)
        return parsed_yaml if isinstance(parsed_yaml, dict) else None
    except Exception:
        return None


def _extract_dataset_meta_from_knowledge(rows: list[dict], dataset_title: str) -> dict[str, str]:
    title_aliases = _build_title_aliases(dataset_title or "")
    if not title_aliases:
        return {"tag": "", "publication": "", "short_description": ""}

    metadata_dict: dict[str, str] = {}
    canonical_refs = {
        "tag",
        "badge",
        "category",
        "type",
        "label",
        "publication",
        "paper",
        "doi",
        "article",
        "short description",
        "shortdescription",
        "description",
    }

    def assign_to_dict(key: str, candidate) -> None:
        normalized_key = _normalize_lookup_key(key)
        if not normalized_key or normalized_key in metadata_dict:
            return
        if candidate is None:
            return
        value = str(candidate).strip()
        if not value:
            return
        metadata_dict[normalized_key] = value

    def matches_title(reference: str, obj: dict | None) -> bool:
        normalized_reference = _normalize_lookup_key(reference)
        compact_reference = normalized_reference.replace(" ", "")
        if any(alias in normalized_reference or alias in compact_reference for alias in title_aliases):
            return True

        if not obj:
            return False

        object_title = str(_first_present(obj, ["title", "db_title", "dataset", "name"]) or "")
        object_aliases = _build_title_aliases(object_title)
        return any(alias in title_aliases for alias in object_aliases)

    for row in rows:
        reference = str(row.get("reference") or "")
        normalized_reference = _normalize_lookup_key(reference)
        obj = _try_parse_structured_object(row.get("content"))
        is_canonical_ref = normalized_reference in canonical_refs
        if not matches_title(reference, obj) and not is_canonical_ref:
            continue

        assign_to_dict(reference, row.get("content"))
        if obj:
            for key, value in obj.items():
                assign_to_dict(str(key), value)

    # Log here if needed: 
    print(f"Extracted metadata for '{dataset_title}': {metadata_dict}")

    def pick_first(keys: list[str]) -> str:
        for key in keys:
            value = metadata_dict.get(_normalize_lookup_key(key), "")
            if value:
                return value
        return ""

    return {
        "tag": pick_first(["tag", "badge", "category", "type", "label"]),
        "publication": pick_first(["publication", "paper", "doi", "article"]),
        "short_description": pick_first(["short_description", "shortDescription", "description"]),
    }


def init_store() -> None:
    with _get_connection() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS database_servers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                db_url TEXT,
                db_hostname TEXT,
                db_port INTEGER,
                db_database TEXT,
                db_user TEXT,
                db_password TEXT,
                db_type TEXT,
                db_schema TEXT,
                db_title TEXT,
                db_tag TEXT,
                db_publication TEXT,
                db_short_description TEXT,
                chat_hostname TEXT,
                chat_port INTEGER,
                chat_database TEXT,
                chat_user TEXT,
                chat_password TEXT,
                chat_schema TEXT,
                UNIQUE(db_title)
            )
            """
        )
        _ensure_database_servers_columns(con)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        con.commit()


def get_active_config_id() -> int | None:
    with _get_connection() as con:
        row = con.execute(
            "SELECT value FROM app_settings WHERE key = 'active_config_id'"
        ).fetchone()
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def set_active_config_id(config_id: int | None) -> None:
    with _get_connection() as con:
        if config_id is None:
            con.execute("DELETE FROM app_settings WHERE key = 'active_config_id'")
        else:
            con.execute(
                "INSERT OR REPLACE INTO app_settings (key, value) VALUES ('active_config_id', ?)",
                (str(config_id),),
            )
        con.commit()


def list_configs() -> list[dict]:
    with _get_connection() as con:
        rows = con.execute(
            """
            SELECT id, db_url, db_hostname, db_port, db_database, db_user, db_password,
                   db_type, db_schema, db_title, db_tag, db_publication, db_short_description,
                   chat_hostname, chat_port, chat_database,
                   chat_user, chat_password, chat_schema
            FROM database_servers
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_config(config_id: int) -> dict | None:
    with _get_connection() as con:
        row = con.execute(
            "SELECT * FROM database_servers WHERE id = ?",
            (config_id,),
        ).fetchone()
    return dict(row) if row else None


def save_config(payload: dict) -> int:
    db_cfg = payload.get("database_server", {})
    chat_cfg = payload.get("chat_database_server", {})
    with _get_connection() as con:
        cur = con.execute(
            """
            INSERT OR IGNORE INTO database_servers
            (db_url, db_hostname, db_port, db_database, db_user, db_password, db_type, db_schema,
             db_title, chat_hostname, chat_port, chat_database, chat_user, chat_password, chat_schema)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                db_cfg.get("url"),
                db_cfg.get("hostname"),
                db_cfg.get("port", 5432),
                db_cfg.get("database"),
                db_cfg.get("user"),
                db_cfg.get("password"),
                db_cfg.get("type", "postgresql"),
                db_cfg.get("schema", "public"),
                db_cfg.get("title", "Untitled"),
                chat_cfg.get("hostname"),
                chat_cfg.get("port", 5432),
                chat_cfg.get("database"),
                chat_cfg.get("user"),
                chat_cfg.get("password"),
                chat_cfg.get("schema", "public"),
            ),
        )
        con.commit()

        if cur.rowcount > 0:
            return int(cur.lastrowid)

        row = con.execute(
            "SELECT id FROM database_servers WHERE db_title = ?",
            (db_cfg.get("title", "Untitled"),),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to save configuration")
        return int(row[0])


def update_config(config_id: int, payload: dict) -> bool:
    db_cfg = payload.get("database_server", {})
    chat_cfg = payload.get("chat_database_server", {})
    with _get_connection() as con:
        try:
            cur = con.execute(
                """
                UPDATE database_servers
                SET db_url = ?,
                    db_hostname = ?,
                    db_port = ?,
                    db_database = ?,
                    db_user = ?,
                    db_password = ?,
                    db_type = ?,
                    db_schema = ?,
                    db_title = ?,
                    chat_hostname = ?,
                    chat_port = ?,
                    chat_database = ?,
                    chat_user = ?,
                    chat_password = ?,
                    chat_schema = ?
                WHERE id = ?
                """,
                (
                    db_cfg.get("url"),
                    db_cfg.get("hostname"),
                    db_cfg.get("port", 5432),
                    db_cfg.get("database"),
                    db_cfg.get("user"),
                    db_cfg.get("password"),
                    db_cfg.get("type", "postgresql"),
                    db_cfg.get("schema", "public"),
                    db_cfg.get("title", "Untitled"),
                    chat_cfg.get("hostname"),
                    chat_cfg.get("port", 5432),
                    chat_cfg.get("database"),
                    chat_cfg.get("user"),
                    chat_cfg.get("password"),
                    chat_cfg.get("schema", "public"),
                    config_id,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: database_servers.db_title" in str(exc):
                raise DuplicateConfigTitleError("Configuration title must be unique") from exc
            raise
        con.commit()
        return cur.rowcount > 0


def delete_config(config_id: int) -> bool:
    with _get_connection() as con:
        cur = con.execute("DELETE FROM database_servers WHERE id = ?", (config_id,))
        con.commit()
        return cur.rowcount > 0


def to_runtime_config(row: dict) -> dict:
    return {
        "database_server": {
            "url": row.get("db_url"),
            "hostname": row.get("db_hostname"),
            "port": row.get("db_port", 5432),
            "database": row.get("db_database"),
            "user": row.get("db_user"),
            "password": row.get("db_password"),
            "type": row.get("db_type", "postgresql"),
            "schema": row.get("db_schema", "public"),
            "title": row.get("db_title", "Untitled"),
        },
        "chat_database_server": {
            "hostname": row.get("chat_hostname"),
            "port": row.get("chat_port", 5432),
            "database": row.get("chat_database"),
            "user": row.get("chat_user"),
            "password": row.get("chat_password"),
            "schema": row.get("chat_schema", "public"),
        },
    }


def sync_metadata_from_knowledge(knowledge_items: list[dict], target_config_id: int | None) -> int:
    if target_config_id is None:
        return 0

    with _get_connection() as con:
        row = con.execute(
            "SELECT id, db_title FROM database_servers WHERE id = ?",
            (target_config_id,),
        ).fetchone()
        if row is None:
            return 0

        record_id = int(row["id"])
        title = str(row["db_title"] or "")
        meta = _extract_dataset_meta_from_knowledge(knowledge_items, title)
        con.execute(
            """
            UPDATE database_servers
            SET db_tag = ?,
                db_publication = ?,
                db_short_description = ?
            WHERE id = ?
            """,
            (
                meta["tag"],
                meta["publication"],
                meta["short_description"],
                record_id,
            ),
        )
        con.commit()
    return 1
