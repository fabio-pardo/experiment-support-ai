from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_DIR = Path("data")  # parent directory to ingest
COLLECTION_NAME = "all-my-documents"  # name for your collection
CHUNK_SIZE = 1200  # characters per chunk
CHUNK_OVERLAP = 150  # overlap between chunks

EXCLUDE_FILES_FROM_INGESTION = [
    Path("data/pdfs/CMDR.pdf")
]  # Example: Exclude the sample ticket PDF

# If you prefer OpenAI embeddings, swap this for OpenAIEmbeddingFunction(...)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
