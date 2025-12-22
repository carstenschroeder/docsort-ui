export ROOT_DIR="./DMS-Root"
export INPUT_DIR="./Input"
export STATE_DIR="./App-State"
export DOCLING_URL="http://192.168.0.168:5001"
export OLLAMA_URL="http://192.168.0.168:11434"
export EMBED_MODEL="nomic-embed-text"

uvicorn app:app --host 127.0.0.1 --port 8000