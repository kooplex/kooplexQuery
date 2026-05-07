from fastapi import APIRouter

from app.services.kooplex_bridge import get_db_chat

router = APIRouter()


@router.get("/examples")
def list_examples() -> dict[str, object]:
    db_chat = get_db_chat()
    keys, data = db_chat.fetch_all_examples()
    items = [dict(zip(keys, row)) for row in data]
    return {"items": items}


@router.post("/examples/{question_id}/validate")
def validate_example(question_id: int) -> dict[str, object]:
    db_chat = get_db_chat()
    db_chat.validate_question(question_id)
    return {"validated": True, "question_id": question_id}


@router.delete("/examples/{question_id}")
def delete_example(question_id: int) -> dict[str, object]:
    db_chat = get_db_chat()
    db_chat.delete_row(question_id)
    return {"deleted": True, "question_id": question_id}
