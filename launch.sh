# export VITE_API_BASE="${REPORT_URL}managedb"
export BACKEND_MOUNT_PATH="${REPORT_URL%/}/managedb"
export VITE_API_BASE="${BACKEND_MOUNT_PATH}"
export VITE_PUBLIC_ORIGIN=$SERVERNAME
export VITE_PROXY_PREFIX=${REPORT_URL}
export VITE_BACKEND_TARGET="http://127.0.0.1:9001"
export KOOPLEX_CONFIG_ENV_PATH="${PWD}/config.env"


tmux new-session -d -s apps
tmux new-window -n backend "cd /v/projects/text2sql/david/kooplexQuery/apps/backend && source .venv/bin/activate && BACKEND_MOUNT_PATH='${BACKEND_MOUNT_PATH}' uvicorn app.main:app --host 0.0.0.0 --port 9001"
tmux new-window -n frontend "cd /v/projects/text2sql/david/kooplexQuery/apps/frontend  && VITE_API_BASE='${VITE_API_BASE}' VITE_PUBLIC_ORIGIN='${VITE_PUBLIC_ORIGIN}' VITE_PROXY_PREFIX='${VITE_PROXY_PREFIX}' VITE_BACKEND_TARGET='${VITE_BACKEND_TARGET}' npm run dev -- --port 9000 --host 0.0.0.0"
