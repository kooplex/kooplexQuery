from pathlib import Path
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, health, metadata, settings, validator, vectorstore
from app.services.config_store import init_store, get_active_config_id, get_config, to_runtime_config
from app.services.runtime_state import runtime_state


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _normalize_mount_path(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip()
    if not normalized or normalized == "/":
        return ""
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized.rstrip("/")


api_app = FastAPI(
    title="KooplexQuery API",
    version="0.1.0",
    description="Backend API for the Vite-based KooplexQuery frontend.",
)

mount_path = _normalize_mount_path(
    os.getenv("BACKEND_MOUNT_PATH") or os.getenv("ROOT_PATH")
)

cors_allow_origins = [
    "http://localhost:9000",
    "http://127.0.0.1:9000",
    "https://k8plex-veo.vo.elte.hu",
]

extra_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if origin.strip()]
for origin in extra_origins:
    if origin not in cors_allow_origins:
        cors_allow_origins.append(origin)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_store()

# Restore last active config from SQLite on startup
_active_id = get_active_config_id()
if _active_id is not None:
    _row = get_config(_active_id)
    if _row is not None:
        _cfg = to_runtime_config(_row)
        _cfg["id"] = _active_id
        runtime_state.active_config = _cfg


api_app.include_router(health.router, tags=["health"])
api_app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
api_app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
api_app.include_router(validator.router, prefix="/api/validator", tags=["validator"])
api_app.include_router(metadata.router, prefix="/api/metadata", tags=["metadata"])
api_app.include_router(vectorstore.router, prefix="/api/vectorstore", tags=["vectorstore"])

if mount_path:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.mount(mount_path, api_app)
else:
    app = api_app
