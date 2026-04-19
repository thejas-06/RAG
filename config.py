import os

# Base Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# AI Model Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile" 

# Text Processing / Chunking Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Search Retrieval Config
DEFAULT_TOP_K = 10
DEFAULT_MIN_SCORE = 0.2
