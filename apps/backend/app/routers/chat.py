from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
import os
import sys
import traceback
from sqlalchemy import text

from app.schemas.chat import (
    CorrectSqlRequest,
    CreateSessionRequest,
    GeneratePlotCodeRequest,
    RunQueryRequest,
    SaveQueryForValidationRequest,
    SaveChatItemRequest,
    SaveQueryRequest,
    StreamChatRequest,
)
from app.services.kooplex_bridge import (
    correct_sql_from_error,
    fetch_chat_history,
    fetch_sessions,
    generate_plot_code,
    get_db_chat,
    get_db_source,
    persist_db_config_env,
    prepare_question_for_validation,
    stream_chat_chunks,
)
from app.services.kooplex_bridge import (
    add_model as add_model_entry,
    delete_model as delete_model_entry,
    list_models as list_model_entries,
)

router = APIRouter()


class AddModelRequest(BaseModel):
    model_name: str
    provider: str | None = None


@router.get("/sessions")
def list_sessions() -> dict[str, object]:
    return {"items": fetch_sessions()}


@router.post("/sessions")
def create_session(payload: CreateSessionRequest) -> dict[str, object]:
    db_chat = get_db_chat()
    session_id = db_chat.new_session(
        username=payload.username,
        email=payload.email,
        label=payload.label,
        meta=payload.meta,
        referenced_session=payload.referenced_session_id,
    )
    return {"session_id": session_id}


@router.get("/sessions/{session_id}/history")
def get_session_history(session_id: int) -> dict[str, object]:
    return {"items": fetch_chat_history(session_id)}


@router.post("/sessions/{session_id}/undo")
def undo_last_session_turn(session_id: int) -> dict[str, object]:
    db_chat = get_db_chat()
    schema_name = db_chat._quote_ident(db_chat.schema)

    q = text(
        f"""
        SELECT id, role, sequence
        FROM {schema_name}.chathistory
        WHERE session_id = :session_id
        ORDER BY sequence ASC, id ASC
        """
    )
    with db_chat.engine.connect() as con:
        rows = con.execute(q, {"session_id": session_id}).mappings().all()

    if not rows:
        return {"deleted": 0, "deleted_ids": []}

    def _is_user(role: object) -> bool:
        normalized = str(role or "").strip().lower()
        return normalized in {"user", "human"}

    def _is_system(role: object) -> bool:
        normalized = str(role or "").strip().lower()
        return normalized == "system"

    last_user_idx = -1
    for idx in range(len(rows) - 1, -1, -1):
        if _is_user(rows[idx].get("role")):
            last_user_idx = idx
            break

    if last_user_idx < 0:
        return {"deleted": 0, "deleted_ids": []}

    deleted_ids: list[int] = [int(rows[last_user_idx]["id"])]

    # Delete the first non-system assistant-like message after the user turn, if present.
    for row in rows[last_user_idx + 1 :]:
        role = row.get("role")
        if _is_system(role):
            continue
        if _is_user(role):
            break
        deleted_ids.append(int(row["id"]))
        break

    id_csv = ",".join(str(value) for value in deleted_ids)
    delete_q = text(
        f"DELETE FROM {schema_name}.chathistory WHERE id IN ({id_csv})"
    )
    with db_chat.engine.begin() as con:
        con.execute(delete_q)

    return {"deleted": len(deleted_ids), "deleted_ids": deleted_ids}


@router.post("/messages")
def save_message_pair(payload: SaveChatItemRequest) -> dict[str, object]:
    db_chat = get_db_chat()
    db_chat.save_chat_item(
        session_id=payload.session_id,
        user_prompt=payload.user_prompt,
        agent_response=payload.agent_response,
        model_name=payload.model_name,
    )
    return {"saved": True}


@router.post("/queries")
def save_query(payload: SaveQueryRequest) -> dict[str, object]:
    db_chat = get_db_chat()
    db_chat.save_query(
        session_id=payload.session_id,
        question_content=payload.question_content,
        sql=payload.sql,
        question_type=payload.question_type,
        public=payload.public,
    )
    return {"saved": True}


@router.post("/queries/prepare-save")
async def save_query_for_validation(payload: SaveQueryForValidationRequest) -> dict[str, object]:
    db_chat = get_db_chat()
    question = (payload.question_content or "").strip()
    if not question:
        question = await prepare_question_for_validation(
            session_id=payload.session_id,
            sql=payload.sql,
            model_name=payload.model_name,
            model_provider=payload.model_provider,
        )

    if payload.preview_only:
        return {"saved": False, "question_content": question}

    db_chat.save_query(
        session_id=payload.session_id,
        question_content=question,
        sql=payload.sql,
        question_type="user",
        public=False,
    )
    return {"saved": True, "question_content": question}


@router.post("/query")
def run_query(payload: RunQueryRequest) -> dict[str, object]:
    try:
        db_source = get_db_source()
        sql = payload.sql.strip().rstrip(";")
        result = db_source.query(sql)
        rows = result.fetchall()
        if hasattr(result, "keys"):
            columns = list(result.keys())
        elif rows and hasattr(rows[0], "_mapping"):
            columns = list(rows[0]._mapping.keys())
        elif rows and isinstance(rows[0], (list, tuple)):
            columns = [f"col_{idx + 1}" for idx in range(len(rows[0]))]
        else:
            columns = []
        return {
            "columns": columns,
            "rows": [list(row) for row in rows],
            "count": len(rows),
        }
    except Exception as exc:
        detail = (
            f"{exc}\n\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=detail) from exc


@router.post("/stream")
async def stream_chat_response(payload: StreamChatRequest) -> StreamingResponse:
    async def generate():
        collected: list[str] = []
        try:
            async for chunk in stream_chat_chunks(
                payload.prompt,
                payload.model_name,
                payload.model_provider,
                payload.session_id,
            ):
                collected.append(chunk)
                yield chunk
        except Exception as exc:
            yield f"\n[stream-error] {exc}"
        finally:
            if payload.session_id is not None and collected:
                try:
                    db_chat = get_db_chat()
                    db_chat.save_chat_item(
                        session_id=payload.session_id,
                        user_prompt=payload.prompt,
                        agent_response="".join(collected),
                        model_name=payload.model_name,
                    )
                except Exception:
                    # Keep streaming path resilient; history can still be saved manually from UI.
                    pass

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@router.get("/models")
def list_models() -> dict[str, object]:
    return {"items": list_model_entries()}


@router.post("/models")
def add_model(payload: AddModelRequest) -> dict[str, object]:
    return {"item": add_model_entry(payload.model_name, payload.provider)}


@router.delete("/models")
def delete_model(model_name: str, provider: str | None = None) -> dict[str, object]:
    return {"deleted": delete_model_entry(model_name, provider)}


class RunCodeRequest(BaseModel):
    code: str
    sql: str | None = None


@router.post("/correct-sql")
async def correct_sql_endpoint(payload: CorrectSqlRequest) -> dict[str, str]:
    return await correct_sql_from_error(
        sql=payload.sql,
        error_message=payload.error_message,
        history_summary=payload.history_summary,
        model_name=payload.model_name,
        model_provider=payload.model_provider,
    )


@router.post("/plot-code")
async def generate_plot_code_endpoint(payload: GeneratePlotCodeRequest) -> dict[str, str]:
    return await generate_plot_code(
        session_id=payload.session_id,
        model_name=payload.model_name,
        model_provider=payload.model_provider,
        sql_override=payload.sql_override,
        plotting_request=payload.plotting_request,
    )


@router.post("/run-code")
def run_code(payload: RunCodeRequest) -> dict[str, object]:
    """Execute Python code and return any Plotly figure as HTML, plus stdout/stderr.
    
    If SQL is provided, it will be executed against the source database and results
    will be available as a pandas DataFrame in the namespace named 'df'.
    """
    code = payload.code
    sql = payload.sql

    # Capture stdout
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    namespace: dict = {}
    plot_html: str | None = None
    error: str | None = None

    try:
        # Persist active DB settings for generated plotting code.
        env_path = persist_db_config_env()
        os.environ["KOOPLEX_CONFIG_ENV_PATH"] = str(env_path)
        namespace["_KOOPLEX_CONFIG_ENV_PATH"] = str(env_path)

        # If SQL is provided, execute it against the source database
        if sql and sql.strip():
            try:
                db_source = get_db_source()
                sql_clean = sql.strip().rstrip(";")
                result = db_source.query(sql_clean)
                rows = result.fetchall()
                
                # Import pandas for DataFrame creation
                import pandas as pd
                
                # Extract column names
                if hasattr(result, "keys"):
                    columns = list(result.keys())
                elif rows and hasattr(rows[0], "_mapping"):
                    columns = list(rows[0]._mapping.keys())
                elif rows and isinstance(rows[0], (list, tuple)):
                    columns = [f"col_{idx + 1}" for idx in range(len(rows[0]))]
                else:
                    columns = []
                
                # Convert rows to list of lists for DataFrame
                data = [list(row) for row in rows]
                df = pd.DataFrame(data, columns=columns)
                namespace["df"] = df
            except Exception as sql_error:
                # If SQL execution fails, report it but continue with code execution
                error = f"SQL execution failed: {traceback.format_exc()}"
                # Return early if we can't execute SQL but it was required
                return {
                    "plot_html": None,
                    "stdout": stdout_buf.getvalue(),
                    "stderr": stderr_buf.getvalue(),
                    "error": error,
                }
        
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buf, stderr_buf
        try:
            exec(compile(code, "<agent-code>", "exec"), namespace)  # noqa: S102
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        # Try to grab a Plotly figure — look for `fig` variable first, then any Figure instance.
        try:
            import plotly.graph_objects as go

            fig = None
            if "fig" in namespace and isinstance(namespace["fig"], go.Figure):
                fig = namespace["fig"]
            else:
                for val in namespace.values():
                    if isinstance(val, go.Figure):
                        fig = val
                        break

            if fig is not None:
                plot_html = fig.to_html(full_html=False, include_plotlyjs=True)
        except ImportError:
            pass  # plotly not installed — no plot output

    except Exception:
        error = traceback.format_exc()

    return {
        "plot_html": plot_html,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "error": error,
    }

