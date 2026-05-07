# KooplexQuery Backend (FastAPI)

This is the new backend for the Vite frontend migration.

## Run locally

```bash
cd apps/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## API docs

- Swagger UI: http://localhost:8080/docs
- OpenAPI JSON: http://localhost:8080/openapi.json

## Scope

Route groups are created for all must-have Streamlit capabilities:

- Settings and database connections
- Chat and sessions
- Validator actions
- Metadata management
- Vectorstore operations

Current implemented groups:

- `/health`
- `/api/settings/*`
- `/api/chat/*`
- `/api/validator/*`
- `/api/metadata/*`
- `/api/vectorstore/*` (requires vectorstore dependencies)
