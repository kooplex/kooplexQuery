from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.kooplex_bridge import get_db_chat, get_db_source, get_vectorstore

router = APIRouter()


class VectorstoreSearchRequest(BaseModel):
    query: str
    collections: list[str] | None = None
    k: int = 5


def _safe_vectorstore():
    try:
        return get_vectorstore()
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vectorstore dependencies are missing: {exc}. Install backend extras first.",
        ) from exc


@router.get("/stats")
def vectorstore_stats() -> dict[str, object]:
    vs = _safe_vectorstore()
    stats = {}
    for coll_name in vs.get_collections():
        coll = vs._init_db(collection_name=coll_name)
        stats[coll_name] = len(coll.get()["ids"])
    return {"collections": stats}


@router.post("/search")
def vectorstore_search(payload: VectorstoreSearchRequest) -> dict[str, object]:
    vs = _safe_vectorstore()
    collections = payload.collections or vs.get_collections()
    out: dict[str, list[dict]] = {}
    for coll_name in collections:
        coll = vs._init_db(collection_name=coll_name)
        results = coll.similarity_search_with_score(payload.query, k=max(1, payload.k))
        out[coll_name] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]
    return {"results": out}


@router.post("/resync")
def vectorstore_resync() -> dict[str, object]:
    vs = _safe_vectorstore()
    db_chat = get_db_chat()
    db_source = get_db_source()

    # Sync validated training examples.
    keys, rows = db_chat.fetch_all_examples()
    examples = [dict(zip(keys, row)) for row in rows]
    for item in examples:
        if item.get("type") == "train" and item.get("public"):
            vs.add_to_examples(
                {
                    "question": item.get("question_content") or "",
                    "sql": item.get("sql") or "",
                }
            )

    # Sync source DB metadata.
    for table_name, table_desc in db_source.describe_tables() or ():
        vs.add_to_docs(
            texts=[str(table_desc or table_name)],
            metadatas=[{"Table": str(table_name), "type": "table_description"}],
        )

    for row in db_source.describe_columns() or ():
        text_value = " - ".join(str(v) for v in row if v is not None)
        vs.add_to_docs(
            texts=[text_value],
            metadatas=[{"Column": str(row[1]), "type": "column_description"}],
        )

    # Sync all knowledge docs.
    k_keys, k_rows = db_chat.fetch_all_knowledge()
    knowledge_items = [dict(zip(k_keys, row)) for row in k_rows]
    for item in knowledge_items:
        vs.add_to_docs(
            texts=[str(item.get("content") or "")],
            metadatas=[{"Reference": str(item.get("reference") or ""), "type": "knowledge"}],
        )

    return {
        "resynced": True,
        "examples": len(examples),
        "knowledge": len(knowledge_items),
    }


@router.post("/reset")
def vectorstore_reset() -> dict[str, object]:
    vs = _safe_vectorstore()
    vs.reset()
    return {"reset": True}
