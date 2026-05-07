from __future__ import annotations

import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "database_servers.sqlite"


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


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
                   db_type, db_schema, db_title, chat_hostname, chat_port, chat_database,
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
