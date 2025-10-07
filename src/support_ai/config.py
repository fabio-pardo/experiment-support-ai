from pathlib import Path

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_DIR = Path("data")  # parent directory to ingest
COLLECTION_NAME = "all-my-documents"  # name for your collection
CHUNK_SIZE = 1200  # characters per chunk
CHUNK_OVERLAP = 150  # overlap between chunks

EXCLUDE_FILES_FROM_INGESTION = [
    Path("data/pdfs/CMDR.pdf")
]  # Example: Exclude the sample ticket PDF

GEMINI_MODEL_NAME = "gemini-2.5-pro"
# If you prefer OpenAI embeddings, swap this for OpenAIEmbeddingFunction(...)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_FN: SentenceTransformerEmbeddingFunction = (
    SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
)
