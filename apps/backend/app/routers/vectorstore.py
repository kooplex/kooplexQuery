from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.kooplex_bridge import (
    ensure_vectorstore_seeded,
    get_vectorstore,
    get_vectorstore_collection_counts,
    sync_vectorstore_from_sources,
)

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
    stats = ensure_vectorstore_seeded(vs)
    return {"collections": stats}


@router.post("/search")
def vectorstore_search(payload: VectorstoreSearchRequest) -> dict[str, object]:
    vs = _safe_vectorstore()
    ensure_vectorstore_seeded(vs)
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
    stats = sync_vectorstore_from_sources(vs)

    return {
        "resynced": True,
        "examples": int(stats.get("examples", 0)),
        "knowledge": int(stats.get("docs", 0)),
        "collections": stats,
    }


@router.post("/reset")
def vectorstore_reset() -> dict[str, object]:
    vs = _safe_vectorstore()
    vs.reset()
    return {"reset": True, "collections": get_vectorstore_collection_counts(vs)}
