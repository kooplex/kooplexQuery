# mcp_server.py
from __future__ import annotations

import os
import logging
import importlib.metadata
import sys, platform
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Any, Dict, List
from textwrap import dedent

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from pathlib import Path



# ---- load env and logging ----------------------------------------------------
load_dotenv()
logger = logging.getLogger("mcp server")
logging.basicConfig(
#    stream=sys.stderr,
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.info("MCP %s | Python %s | %s", importlib.metadata.version("mcp"), sys.version.split()[0], platform.platform())

#try:
#    from .db import DBQuery
#except:
#    logging.critical("cannot import DBQuery")
#    DBQuery = None
# Helpful breadcrumbs while debugging:
logging.debug("SERVER start: file=%s", __file__)
logging.debug("initial sys.path[0]=%s", sys.path[0])

DBQ_IMPORT_ERR = None
try:
    # Case A: launched as a module (python -m mcp_server.entry)
    from .db import DBQuery  # package-relative
    logging.debug("Imported DBQuery via relative import (.db)")
except Exception as e1:
    DBQ_IMPORT_ERR = e1
    try:
        # Case B: repo root on PYTHONPATH (from client)
        from mcp_server.db import DBQuery  # absolute package import
        logging.debug("Imported DBQuery via absolute import (mcp_server.db)")
    except Exception as e2:
        try:
            # Case C: launched by file path (mcp dev entry.py), no package context
            here = Path(__file__).resolve().parent
            sys.path.insert(0, str(here))  # add server directory itself
            from db import DBQuery  # plain module import in same folder
            logging.debug("Imported DBQuery via local module import (db)")
        except Exception as e3:
            logging.exception(
                "Failed to import DBQuery. "
                "Errors were:\n - relative: %r\n - absolute: %r\n - local: %r",
                DBQ_IMPORT_ERR, e2, e3
            )
            raise  # stop early; don’t run without DBQuery
# ---- end shim ----


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
            logger.debug("DB context prepared")
        except Exception:
            logger.exception("DB init failed; continuing without DB")

    try:
        #yield AppCtx(db=db)
        logger.debug("STARTUP dbq=%s pool=%s id(dbq)=%s id(pool)=%s",
            db, getattr(db, "_pool", None), id(db), id(getattr(db, "_pool", None)))
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

##### ---- Resources ---------------------------------------------------------------
@mcp.resource("doc://viralprimer")
def viralprimer_doc() -> str:
    """
    Comprehensive resources for monitoring SARS-CoV-2 primer efficiency: 
    mutation datasets and the ViralPrimer web server
    """
    resource_path = Path(__file__).parent / "resources" / "viralprimer_description.txt"
    try:
        return resource_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[Error loading resource: {e}]"

# good to know we could do it dynamically
#for path in Path("resources").glob("*.txt"):
#    @mcp.resource(path.stem)
#    def make_resource(ctx: Context, p=path):
#        return p.read_text(encoding="utf-8")


##### ---- Prompts -----------------------------------------------------------------
@mcp.prompt("sql_generation")
#def sql_generation_prompt() -> list[PromptMessage]:
def sql_generation_prompt() -> str:
    """
    You are a SQL expert. Generate PostgreSQL queries against the given schema.
    Use explicit schema qualification and only read data.
    """
    return dedent("""
        You are a SQL expert. Generate PostgreSQL queries against the given schema.
        - Use explicit schema qualification.
        - Read-only: do not modify data (no INSERT/UPDATE/DELETE/DDL).
        - Prefer JOINs on primary/foreign keys. Return concise queries.
    """).strip()


##### ---- Entrypoint --------------------------------------------------------------

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()

