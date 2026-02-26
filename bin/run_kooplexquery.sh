SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$SOURCE")"

streamlit run $SCRIPT_DIR/..//kooplexQuery/prompt.py --server.baseUrlPath=$REPORT_URL --server.port=9000 --server.address=0.0.0.0 --server.runOnSave=True --server.allowRunOnSave=True \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.enableWebsocketCompression false \
    --browser.gatherUsageStats false \
    --logger.level debug \
    --server.headless true
