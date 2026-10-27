

import os

MODEL_PROVIDER = "ollama"

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL    = "gemini-3.8-flash"

OLLAMA_URL      = "http://localhost:11434"
OLLAMA_MODEL    = "gemma4:e4b"

EMBEDDING_MODEL = "nomic-embed-text:latest"

# os.getenv("GEMINI_MODEL","gemini-3.8-flash")