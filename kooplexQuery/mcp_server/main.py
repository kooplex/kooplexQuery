# mcp_server.py
from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Any, Dict, List

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv


try:
    from db import DBQuery
except:
    DBQuery = None


# ---- load env and logging ----------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---- MCP app setup with DI and lifecycle ------------------------------------
# when mpi dev main.py error is raised
#@dataclass
#class AppCtx:
#    db: object | None #FIXME keep loose while learning; tighten later

@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppCtx]:
    # Build DSN from env
    PG_HOST=os.getenv('PG_HOST', 'localhost')
    PG_PORT=os.getenv('PG_PORT', 5432)
    PG_USER=os.getenv('PG_USER')
    PG_PASSWORD=os.getenv('PG_PASSWORD', '')
    PG_DATABASE=os.getenv('PG_DATABASE', '')
    PG_SCHEMA=os.getenv('PG_SCHEMA', 'public')
    dsn = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
    # Mask password in logs
    safe_dsn = dsn.replace(PG_PASSWORD, "****") if PG_PASSWORD else dsn
    logger.info("Connecting to DB (DSN: %s)...", safe_dsn)
#    db: Optional[Any] = None
    db: Any | None = None
    if DBQuery is not None and PG_USER and PG_DATABASE:
        try:
            db = DBQuery(dsn, PG_SCHEMA, timeout_s=30)
            await db.connect()
            logger.info("DB context prepared")
        except Exception:
            logger.exception("DB init failed; continuing without DB")

    try:
        #yield AppCtx(db=db)
        yield {"db": db}
    finally:
        # Clean shutdown
        if db and hasattr(db, "close"):
            try:
                maybe_close = db.close()
                # support async or sync close
                if hasattr(maybe_close, "__await__"):
                    await maybe_close
                logger.info("DB connection closed")
            except Exception:
                logger.exception("Error during DB close")


mcp = FastMCP("db_viralprimer", lifespan=lifespan)

# ---- a simple test tool so you can verify wiring -----------------------------
class PingInput(BaseModel):
    msg: str | None = Field(None, description="Optional message to echo")

class PingOutput(BaseModel):
    pong: str
    has_db: bool
    echo: str | None = None

@mcp.tool(description="Simple liveness check.")
async def ping(ctx: Context, payload: PingInput) -> PingOutput:
#    app: AppCtx = ctx.request_context.lifespan_context  # proves lifespan ran
    app = ctx.request_context.lifespan_context  # proves lifespan ran
#    has_db = bool(app and app.db)
    has_db = bool(app.get("db"))
    await ctx.info(f"ping called | has_db={has_db} | echo={payload.msg!r}")
    return PingOutput(
        pong="ok",
        has_db=has_db,
        echo=payload.msg,
    )


class RunInput(BaseModel):
#    sql: str
#    params: Dict[str, Any] | List[Any] | None = None
#    fetch: Literal["none","one","all"] = "all"
    sql: str
    params: object = Field(None, description="Either a dict for named params or a list/tuple for positional params")
    fetch: str = Field("all", description="one of: none, one, all")

    @field_validator("fetch")
    @classmethod
    def _fetch_ok(cls, v: str) -> str:
        if v not in ("none", "one", "all"):
            raise ValueError("fetch must be one of: none, one, all")
        return v

    @field_validator("params")
    @classmethod
    def _params_ok(cls, v):
        if v is None:
            return None
        if isinstance(v, (dict, list, tuple)):
            return v
        raise TypeError("params must be a dict, list, tuple, or null")

class RunOutput(BaseModel):
#    result: Any
    result: object

@mcp.tool(description="Execute SQL (demo).")
async def run(ctx: Context, payload: RunInput) -> RunOutput:
    app = ctx.request_context.lifespan_context
    db = app.get("db")
    if not db:
        return RunOutput(result={"error": "DB not initialized"})
    try:
        res = await db.run_query(payload.sql, payload.params, payload.fetch)
        return RunOutput(result=res)
    except Exception as e:
        await ctx.error(f"run failed: {e!r}")
        return RunOutput(result={"error": str(e)})


class DescribeTableInput(BaseModel):
    table_name_like: str | None

class DescribeTableOutput(BaseModel):
    result: object

@mcp.tool(description="Describe tables, whose name match the pattern")
async def describe_tables(ctx: Context, payload: DescribeTableInput) -> DescribeTableOutput:
    app = ctx.request_context.lifespan_context
    db = app.get("db")
    if not db:
        return DescribeTableOutput(result={"error": "DB not initialized"})
    try:
        res = await db.describe_tables(payload.table_name_like)
        return DescribeTableOutput(result=res)
    except Exception as e:
        await ctx.error(f"Looking up table description failed: {e!r}")
        return DescribeTableOutput(result={"error": str(e)})


class DescribeColumnInput(BaseModel):
    table_name_like: str | None

class DescribeColumnOutput(BaseModel):
    result: object

@mcp.tool(description="Describe table columns, whose table name match the pattern")
async def describe_columns(ctx: Context, payload: DescribeColumnInput) -> DescribeColumnOutput:
    app = ctx.request_context.lifespan_context
    db = app.get("db")
    if not db:
        return DescribeColumnOutput(result={"error": "DB not initialized"})
    try:
        res = await db.describe_columns(payload.table_name_like)
        return DescribeColumnOutput(result=res)
    except Exception as e:
        await ctx.error(f"Looking up table description failed: {e!r}")
        return DescribeColumnOutput(result={"error": str(e)})


class DDLInput(BaseModel):
    kind: str = Field(..., description="One of: tables, views, types, functions")

class DDLOutput(BaseModel):
    result: object

@mcp.tool(description="Return reconstructed CREATE statements for schema objects.")
async def ddl_extract(ctx: Context, payload: DDLInput) -> DDLOutput:
    app = ctx.request_context.lifespan_context
    db = app.get("db")
    if not db:
        return DDLOutput(result={"error": "DB not initialized"})

    try:
        kind = payload.kind.lower()
        if kind == "tables":
            rows = await db.DDL_tables()
        elif kind == "views":
            rows = await db.DDL_views()
        elif kind == "types":
            rows = await db.DDL_types()
        elif kind == "functions":
            rows = await db.DDL_functions()
        else:
            return DDLOutput(result={"error": f"Unknown kind: {payload.kind}"})
        return DDLOutput(result=rows)
    except Exception as e:
        await ctx.error(f"DDL extract failed: {e!r}")
        return DDLOutput(result={"error": str(e)})


##### ---- Entrypoint --------------------------------------------------------------

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()

