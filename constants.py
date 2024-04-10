import os
from chromadb.config import Settings
from langchain.document_loaders import UnstructuredPDFLoader
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
Category = "JAVA"
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/DATA/{Category}"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB/{Category}"

MODELS_PATH = "./models"

INGEST_THREADS = os.cpu_count() or 8

CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
    is_persistent=True,
)

CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE/4


N_GPU_LAYERS = 100
N_BATCH = 512
DOCUMENT_MAP = {
    ".pdf": UnstructuredPDFLoader,
}
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"

MODEL_ID = "TheBloke/Vigogne-2-7B-Chat-GGUF"
MODEL_BASENAME = "vigogne-2-7b-chat.Q4_K_M.gguf"
