# chroma.py
import re
from pathlib import Path
from typing import List, Tuple, Dict
from pypdf import PdfReader
import webvtt

import chromadb
from config import (
    DATA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    embedding_fn,
    EXCLUDE_FILES_FROM_INGESTION,
)


# ---- Helpers ----
def read_pdf_content(p: Path) -> str:
    text_parts = []
    with p.open("rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            # ensure text is a string (some pages can return None)
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def read_vtt_content(p: Path) -> str:
    """
    Use webvtt if available; fall back to a simple parser.
    """
    try:
        return "\n".join([c.text for c in webvtt.read(str(p))]).strip()
    except Exception:
        # very light-weight VTT fallback (ignores timing)
        content = p.read_text(encoding="utf-8", errors="ignore")
        # remove WEBVTT header and timestamps
        content = re.sub(r"^WEBVTT.*?$", "", content, flags=re.MULTILINE)
        content = re.sub(r"\d+\n", "", content)
        content = re.sub(
            r"\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}.*", "", content
        )
        return re.sub(r"\n{2,}", "\n", content).strip()


def read_text_file_content(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def extract_document_data(p: Path) -> Tuple[str, Dict]:
    """
    Returns (text, metadata) for supported files.
    """
    ext = p.suffix.lower()
    if ext == ".pdf":
        text = read_pdf_content(p)
        doc_type = "pdf"
    elif ext in {".vtt", ".srt"}:
        text = read_vtt_content(p)
        doc_type = ext.lstrip(".")
    elif ext in {".txt", ".md", ".rst"}:
        text = read_text_file_content(p)
        doc_type = "text"
    elif ext in {".py", ".sh"}:
        text = read_text_file_content(p)
        doc_type = "code"
    elif ext in {".mp4", ".mov", ".mkv"}:
        # Prefer a sibling transcript if present; otherwise skip video binary
        for sidecar in [p.with_suffix(".vtt"), p.with_suffix(".srt")]:
            if sidecar.exists():
                text = extract_document_data(sidecar)[0]
                doc_type = "video_transcript"
                break
        else:
            return "", {}
    else:
        # unsupported file type
        return "", {}

    meta = {
        "source": str(p),
        "name": p.name,
        "parent": str(p.parent),
        "type": doc_type,
    }
    return text, meta


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i : min(i + size, n)]
        chunks.append(chunk.strip())
        if i + size >= n:
            break
        i += size - overlap
    return [c for c in chunks if c]


def prepare_ingestion_chunks_from_directory(root: Path) -> List[Tuple[str, Dict, str]]:
    """
    Walks DATA_DIR and returns a list of (chunk_text, metadata, id)
    """
    items: List[Tuple[str, Dict, str]] = []
    processed_transcripts = (
        set()
    )  # Track which transcript files we've processed via videos

    # First pass to collect all video files and their transcripts
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in p.parts):
            continue

        # If it's a video, check for transcripts and mark them
        if p.suffix.lower() in {".mp4", ".mov", ".mkv"}:
            for sidecar in [p.with_suffix(".vtt"), p.with_suffix(".srt")]:
                if sidecar.exists():
                    processed_transcripts.add(sidecar)

    # Second pass to process files while skipping redundant transcripts
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in p.parts):
            continue

        # Skip transcript files we've already processed via their videos
        if p in processed_transcripts or p in EXCLUDE_FILES_FROM_INGESTION:
            continue

        text, meta = extract_document_data(p)
        if not text:
            continue

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            rel = p.relative_to(root.parent) if root in p.parents else p
            doc_id = f"{rel.as_posix()}::chunk-{idx:04d}"
            items.append((chunk, meta, doc_id))
    return items


# ---- Main ingest ----
def run_document_ingestion():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    items = prepare_ingestion_chunks_from_directory(DATA_DIR)
    if not items:
        print(f"No ingestible files found under {DATA_DIR.resolve()}")
        return

    # Get existing IDs from collection
    try:
        existing_ids = set(collection.get()["ids"])
    except Exception:
        existing_ids = set()

    # Filter out items that already exist
    new_items = [
        (doc, meta, _id) for doc, meta, _id in items if _id not in existing_ids
    ]

    if not new_items:
        print("No new documents to index")

    print(f"Skipping {len(items) - len(new_items)} existing documents")
    print(f"Indexing {len(new_items)} new documents...")

    # Process only new items
    BATCH = 100
    docs, metas, ids = [], [], []
    for i, (doc, meta, _id) in enumerate(new_items, 1):
        docs.append(doc)
        metas.append(meta)
        ids.append(_id)

        if len(docs) == BATCH or i == len(new_items):
            collection.add(documents=docs, metadatas=metas, ids=ids)
            print(f"Indexed {len(ids)} / {len(new_items)} new chunks...")
            docs, metas, ids = [], [], []

    print(
        f"✅ Ingest complete. {collection.count()} total records in '{COLLECTION_NAME}'."
    )
