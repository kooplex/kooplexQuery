# db.py
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Literal, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from psycopg import sql as psql

# ---- Public errors your server can map to ToolError cleanly ------------------

class DBQueryError(Exception): ...
class DBTimeoutError(DBQueryError): ...
class DBNotFound(DBQueryError): ...
class DBConnectionError(DBQueryError): ...

# ---- The async DB client -----------------------------------------------------

class DBQuery:
    """
    Minimal async Postgres client with a pooled connection and a single
    run_query entrypoint.

    Param style:
      - Use %(name)s for named params, or %s for positional.
        e.g. "SELECT * FROM t WHERE id = %(id)s", {"id": 1}
             "INSERT INTO t(a,b) VALUES (%s,%s)", [10, "x"]
    """
    def __init__(
        self,
        dsn: str,
        schema: Optional[str] = None,
        *,
        timeout_s: int = 30,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._schema = schema
        self._timeout_s = int(timeout_s)
        self._pool_min = pool_min_size
        self._pool_max = pool_max_size
        self._pool: Optional[AsyncConnectionPool] = None

    # -- lifecycle -------------------------------------------------------------

    async def connect(self) -> None:
        """
        Create (but don't auto-open) the async pool, then open it explicitly. Autocommit is enabled for simplicity.
        """
        try:
            self._pool = AsyncConnectionPool(
                conninfo=self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
                kwargs={"autocommit": True},
                open=False,
            )
            # Explicitly open the pool so connection errors show up now
            await self._pool.open()

            # Optionally set search_path for all pooled connections
            if self._schema:
                async with self._pool.connection() as ac:
                    # Safe identifier quoting
                    q = psql.SQL("set search_path to {}").format(psql.Identifier(self._schema))
                    await ac.execute(q)

        except psycopg.OperationalError as e:
            raise DBConnectionError(f"Failed to connect: {e}") from e
        except Exception as e:
            # Surface unexpected issues (bad DSN, auth, etc.)
            raise DBConnectionError(f"Unexpected connect error: {e}") from e

    async def close(self) -> None:
        """
        Close the pool and all its connections.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # -- main API --------------------------------------------------------------

    async def run_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any] | List[Any]] = None,
        fetch: Literal["none", "one", "all"] = "all",
    ) -> Any:
        """
        Execute SQL with optional params.

        Returns:
          - fetch="none": affected rowcount (int or None if unknown)
          - fetch="one" : dict row or None
          - fetch="all" : list[dict]
        """
        pool = self._require_pool()

        try:
            async with asyncio.timeout(self._timeout_s):  # Python 3.11+
                # Each acquisition gets a fresh connection from the pool
                async with pool.connection() as ac:
                    # Row factory returns dicts: {col: value}
                    async with ac.cursor(row_factory=dict_row) as cur:
                        await cur.execute(sql, params)

                        if fetch == "none":
                            # For DML, cur.rowcount is affected rows; for SELECT often -1
                            return None if cur.rowcount == -1 else int(cur.rowcount)

                        if fetch == "one":
                            row = await cur.fetchone()
                            # already a dict due to dict_row
                            return row

                        # "all"
                        rows = await cur.fetchall()
                        return rows

        except asyncio.TimeoutError as e:
            raise DBTimeoutError(f"Query timed out after {self._timeout_s}s") from e

        except psycopg.OperationalError as e:
            # Connection lost, too many connections, etc.
            raise DBConnectionError(str(e)) from e

        except psycopg.errors.UndefinedTable as e:
            raise DBNotFound("Table not found") from e

        except psycopg.Error as e:
            # Any other driver error -> generic query error
            raise DBQueryError(str(e)) from e

        except Exception as e:
            # Unexpected runtime error
            raise DBQueryError(f"Unexpected error: {e}") from e


    async def describe_tables(self, table_name_like: str = None, case_insensitive: bool = True) -> Any:
        """
        List user tables/views with descriptions in the configured schema, optionally filtering by a LIKE pattern.
        - Use % wildcards in table_name_like (e.g., '%user%'); if you pass a plain term,
          we'll wrap it as '%term%'.
        """
        op = "ILIKE" if case_insensitive else "LIKE"
        if table_name_like:
            # Wrap with wildcards unless the caller already provided some
            pattern = table_name_like
            if "%" not in pattern and "_" not in pattern:
                pattern = f"%{pattern}%"
            q=f"""
SELECT c.relname AS table_name, obj_description(c.oid) AS table_comment
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind in ('r', 'v')
  AND n.nspname = %(schema)s 
  AND c.relname {op} %(pattern)s
ORDER BY c.relname;
            """
            return await self.run_query(sql=q, params={'schema': self._schema, 'pattern': pattern})
        q="""
SELECT c.relname AS table_name, obj_description(c.oid) AS table_comment
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind in ('r', 'v')
 AND n.nspname = %(schema)s
ORDER BY c.relname;
        """
        return await self.run_query(sql=q, params={'schema': self._schema})

    async def describe_columns(self, table_name_like: str = None, case_insensitive: bool = True) -> Any:
        """
        List user tables/views with column descriptions in the configured schema, optionally filtering by a LIKE pattern.
        - Use % wildcards in table_name_like (e.g., '%user%'); if you pass a plain term,
          we'll wrap it as '%term%'.
        """
        op = "ILIKE" if case_insensitive else "LIKE"
        if table_name_like:
            # Wrap with wildcards unless the caller already provided some
            pattern = table_name_like
            if "%" not in pattern and "_" not in pattern:
                pattern = f"%{pattern}%"
            q=f"""
SELECT c.relname AS table_name, 
       a.attname AS column_name, 
       col_description(a.attrelid, a.attnum) AS description, 
       pg_catalog.format_type(a.atttypid, a.atttypmod) AS datatype
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE c.relkind in ('r', 'v')
  AND n.nspname = %(schema)s
  AND a.attnum > 0 
  AND NOT a.attisdropped 
  AND c.relname {op} %(pattern)s
ORDER BY c.relname, a.attnum;
            """
            return await self.run_query(sql=q, params={'schema': self._schema, 'pattern': pattern})
        q="""
SELECT c.relname AS table_name, 
       a.attname AS column_name, 
       col_description(a.attrelid, a.attnum) AS description, 
       pg_catalog.format_type(a.atttypid, a.atttypmod) AS datatype
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE c.relkind in ('r', 'v')
  AND n.nspname = %(schema)s
  AND a.attnum > 0 
  AND NOT a.attisdropped
ORDER BY c.relname, a.attnum;
        """
        return await self.run_query(sql=q, params={'schema': self._schema})

    async def DDL_tables(self) -> Any:
        """Return CREATE TABLE statements for all regular tables in the current schema."""
        q="""
SELECT
    '--- TABLE: ' || c.relname AS ddl_header,
    'CREATE TABLE ' || n.nspname || '.' || c.relname || E' (\n' ||
    string_agg(
        '    ' || a.attname || ' ' || pg_catalog.format_type(a.atttypid, a.atttypmod),
        E',\n'
        ORDER BY a.attnum
    ) || E'\n);' AS ddl
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE n.nspname = %(schema)s
  AND c.relkind = 'r'
  AND a.attnum > 0 AND NOT a.attisdropped
GROUP BY c.oid, c.relname, n.nspname
ORDER BY c.relname;
        """
        return await self.run_query(sql=q, params={'schema': self._schema})

    async def DDL_views(self) -> Any:
        """Return CREATE VIEW statements for all views in the current schema."""
        q="""
SELECT '--- VIEW: ' || c.relname AS ddl_header,
       'CREATE VIEW ' || n.nspname || '.' || c.relname || ' AS ' || pg_get_viewdef(c.oid, true) || ';'
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %(schema)s 
  AND c.relkind = 'v';
        """
        return await self.run_query(sql=q, params={'schema': self._schema})

    async def DDL_types(self) -> Any:
        """Return CREATE TYPE statements for all composite and enum types in the current schema."""
        q="""
SELECT '--- TYPE: ' || t.typname AS ddl_header,
       'CREATE TYPE ' || n.nspname || '.' || t.typname || ' AS (' ||
       string_agg(a.attname || ' ' || pg_catalog.format_type(a.atttypid, a.atttypmod), ', ') || ');'
FROM pg_type t
JOIN pg_namespace n ON n.oid = t.typnamespace
JOIN pg_class c ON c.oid = t.typrelid
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE n.nspname = %(schema)s
  AND t.typtype = 'c'
  AND a.attnum > 0 AND NOT a.attisdropped
GROUP BY t.typname, n.nspname
UNION
SELECT '--- ENUM TYPE: ' || t.typname AS ddl_header,
       'CREATE TYPE ' || n.nspname || '.' || t.typname || ' AS ENUM (' ||
       string_agg(quote_literal(e.enumlabel), ', ') || ');'
FROM pg_type t
JOIN pg_enum e ON e.enumtypid = t.oid
JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = %(schema)s
GROUP BY t.typname, n.nspname;
        """
        return await self.run_query(sql=q, params={'schema': self._schema})

    async def DDL_functions(self) -> Any:
        """Return CREATE FUNCTION and CREATE AGGREGATE statements for all routines in the current schema."""
        q="""
SELECT
    '--- FUNCTION: ' || p.proname || '(' || pg_get_function_arguments(p.oid) || ')' AS ddl_header,
    pg_get_functiondef(p.oid) AS ddl
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = %(schema)s
  AND p.prokind = 'f'  -- 'f' = function, 'a' = aggregate, 'p' = procedure, 'w' = window
UNION
SELECT
    '--- AGGREGATE: ' || p.proname || '(' || pg_get_function_arguments(p.oid) || ')' AS ddl_header,
    'CREATE AGGREGATE ' || n.nspname || '.' || p.proname || '(' ||
        pg_get_function_arguments(p.oid) || ') ' ||
        '(SFUNC = ' || sfunc.proname || ', STYPE = ' || format_type(aggtranstype, NULL) || ')' AS ddl
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
JOIN pg_aggregate a ON a.aggfnoid = p.oid
JOIN pg_proc sfunc ON sfunc.oid = a.aggtransfn
WHERE n.nspname = %(schema)s;
        """
        return await self.run_query(sql=q, params={'schema': self._schema})

    # -- helpers ---------------------------------------------------------------

    def _require_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise DBConnectionError("DB pool is not initialized. Call connect() first.")
        return self._pool

