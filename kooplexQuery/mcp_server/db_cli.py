# cli.py
from __future__ import annotations

import os
import json
import argparse
import asyncio
from typing import Any, Dict, List

from db import DBQuery

def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v is not None else default

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DBQuery async CLI")
    p.add_argument("-H", "--server", default=_env("DB_HOST", "localhost"))
    p.add_argument("-P", "--port", default=_env("DB_PORT", "5432"))
    p.add_argument("-D", "--database", default=_env("DB"))
    p.add_argument("-s", "--schema", default=_env("DB_SCHEMA"))
    p.add_argument("-u", "--user", default=_env("SECRET_USERNAME"))
    p.add_argument("-p", "--password", default=_env("SECRET_PASSWORD"))

    p.add_argument("--json", action="store_true", help="Emit JSON")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Run query")
    q.add_argument("-S", "--sql", required=True, help="SQL to execute")
    q.add_argument("--one", action="store_true", help="fetch one row instead of all")

    sub.add_parser("tables", help="List tables with descriptions")

    c = sub.add_parser("columns", help="List columns for a table")
    c.add_argument("--table", required=True, help="Table name")

    ddl = sub.add_parser("ddl", help="Generate DDL code")
    ddl.add_argument("-o", "--object", required=True,
                     choices=["table", "view", "type", "function"],
                     help="Which kind of object DDL to emit")

    return p

def _print(obj: Any, as_json: bool, pretty: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2 if pretty else None, ensure_ascii=False))
    else:
        # best-effort text output
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            # if looks like rows with ddl_header/ddl
            if "ddl" in obj[0]:
                for row in obj:
                    header = row.get("ddl_header") or ""
                    ddl = row.get("ddl") or ""
                    print(header)
                    print(ddl)
                    print()
                return
        print(obj)

async def main() -> None:
    args = build_parser().parse_args()

    # Build DSN
    dsn = f"postgresql://{args.user}:{args.password}@{args.server}:{args.port}/{args.database}"

    db = DBQuery(dsn=dsn, schema=args.schema, timeout_s=30)

    try:
        await db.connect()

        if args.command == "query":
            fetch = "one" if args.one else "all"
            rows = await db.run_query(sql=args.sql, params=None, fetch=fetch)
            _print(rows, args.json, args.pretty)

        elif args.command == "tables":
            rows = await db.describe_tables()        # returns list of {table_name, table_comment}
            _print(rows, args.json, args.pretty)

        elif args.command == "columns":
            # assuming you will implement describe_columns(table: str)
            rows = await db.describe_columns(args.table)
            _print(rows, args.json, args.pretty)

        elif args.command == "ddl":
            kind = args.object
            if kind == "table":
                rows = await db.DDL_tables()
            elif kind == "view":
                rows = await db.DDL_views()
            elif kind == "type":
                rows = await db.DDL_types()
            else:  # "function"
                rows = await db.DDL_functions()
            _print(rows, args.json, args.pretty)

    except Exception as e:
        # simple CLI-friendly error
        print(f"ERROR: {e}")
        raise
    finally:
        try:
            await db.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())

