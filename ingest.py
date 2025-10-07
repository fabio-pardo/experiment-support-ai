# chroma.py
import re
from pathlib import Path  # Ensure Path is imported
from pypdf import PdfReader
import webvtt  # pyright: ignore[reportMissingTypeStubs]

import chromadb
from config import (
    DATA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EXCLUDE_FILES_FROM_INGESTION,
    EMBEDDING_FN,
)


# ---- Helpers ----
def read_pdf_content(p: Path) -> str:
    text_parts: list[str] = []
    with p.open("rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            # ensure text is a string (some pages can return None)
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def process_ticket_pdf(ticket_pdf_path: Path) -> str:
    """
    Reads the content of a PDF support ticket.
    """
    if not ticket_pdf_path.exists():
        return f"Error: Ticket PDF not found at {ticket_pdf_path}"
    if ticket_pdf_path.suffix.lower() != ".pdf":
        return f"Error: File {ticket_pdf_path} is not a PDF."

    print(f"Reading content from {ticket_pdf_path}...")
    ticket_text = read_pdf_content(ticket_pdf_path)
    print("Successfully read PDF content.")
    return ticket_text


def _read_vtt_captions(p: Path) -> list[dict[str, str | None]]:
    """
    Reads a VTT/SRT file and returns a list of dictionaries,
    each containing the caption text, start time, and end time.
    Returns None for start/end if webvtt parsing fails.
    """
    captions_data: list[dict[str, str | None]] = []
    try:
        captions: webvtt.WebVTT = webvtt.read(str(p))
        for caption in captions:  # pyright: ignore[reportUnknownVariableType]
            captions_data.append(
                {
                    "text": caption.text.strip(),  # pyright: ignore[reportUnknownMemberType]
                    "start_time": caption.start,  # e.g., "00:00:01.000"  # pyright: ignore[reportUnknownMemberType]
                    "end_time": caption.end,  # e.g., "00:00:03.500"  # pyright: ignore[reportUnknownMemberType]
                }
            )
    except Exception as e:
        print(
            f"Warning: Could not parse VTT/SRT with webvtt, falling back to simple text extraction for {p}: {e}"
        )
        # Fallback to simple text extraction if webvtt fails, but without timestamps
        content = p.read_text(encoding="utf-8", errors="ignore")
        # Remove common VTT/SRT headers and timestamps from content for clean text
        content = re.sub(r"^WEBVTT.*?$", "", content, flags=re.MULTILINE)
        content = re.sub(
            r"^\d+\s*$", "", content, flags=re.MULTILINE
        )  # Remove cue numbers
        content = re.sub(
            r"\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}.*", "", content
        )  # Remove time ranges
        content = re.sub(
            r"\n{2,}", "\n", content
        ).strip()  # Consolidate multiple newlines

        if content:
            captions_data.append(
                {"text": content, "start_time": None, "end_time": None}
            )
    return captions_data


def read_text_file_content(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


# --- Document Type Handlers ---
def _handle_pdf(p: Path) -> tuple[str, str]:
    return read_pdf_content(p), "pdf"


def _handle_caption(p: Path) -> tuple[list[dict[str, str | None]], str]:
    return _read_vtt_captions(p), "video_caption"


def _handle_text(p: Path) -> tuple[str, str]:
    return read_text_file_content(p), "text"


def _handle_code(p: Path) -> tuple[str, str]:
    return read_text_file_content(p), "code"


def _handle_video(
    p: Path,
) -> tuple[str | list[dict[str, str | None]], str]:
    for sidecar in [p.with_suffix(".vtt"), p.with_suffix(".srt")]:
        if sidecar.exists():
            return _read_vtt_captions(sidecar), "video_transcript"
    return "", "video"


# --- File Extension to Handler Mapping ---
DOCUMENT_HANDLERS = {
    ".pdf": _handle_pdf,
    ".vtt": _handle_caption,
    ".srt": _handle_caption,
    ".txt": _handle_text,
    ".md": _handle_text,
    ".rst": _handle_text,
    ".py": _handle_code,
    ".sh": _handle_code,
    ".mp4": _handle_video,
    ".mov": _handle_video,
    ".mkv": _handle_video,
}


def extract_document_data(
    p: Path,
) -> tuple[str | list[dict[str, str | None]], dict[str, str]]:
    """
    Returns (text, metadata) for supported files by dispatching to a handler.
    """
    ext = p.suffix.lower()
    handler = DOCUMENT_HANDLERS.get(ext)

    if not handler:
        return "", {}  # Unsupported file type

    content, doc_type = handler(p)

    if not content:
        return "", {}

    base_meta = {
        "source": str(p),
        "name": p.name,
        "parent": str(p.parent),
        "type": doc_type,
    }
    return content, base_meta


def chunk_text(
    text: tuple[str | list[dict[str, str | None]]],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str | list[dict[str, str | None]], ...]]:
    chunks: list[tuple[str | list[dict[str, str | None]], ...]] = []
    i = 0
    n = len(text)
    while i < n:
        chunk: tuple[str | list[dict[str, str | None]], ...] = text[
            i : min(i + size, n)
        ]
        chunks.append(chunk)
        if i + size >= n:
            break
        i += size - overlap
    return [c for c in chunks if c]


def prepare_ingestion_chunks_from_directory(
    root: Path,
) -> list[tuple[tuple[str | list[dict[str, str | None]], ...], dict[str, str], str]]:
    """
    Walks DATA_DIR and returns a list of (chunk_text, metadata, id)
    """
    items: list[
        tuple[tuple[str | list[dict[str, str | None]], ...], dict[str, str], str]
    ] = []
    processed_transcripts: set[Path] = (
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

        chunks = chunk_text((text,))
        for idx, chunk in enumerate(chunks):
            rel = p.relative_to(root.parent) if root in p.parents else p
            doc_id = f"{rel.as_posix()}::chunk-{idx:04d}"
            items.append((chunk, meta, doc_id))
    return items


# ---- Main ingest ----
def run_document_ingestion():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDING_FN,  # pyright: ignore[reportArgumentType]
    )

    items = prepare_ingestion_chunks_from_directory(DATA_DIR)
    if not items:
        print(f"No ingestible files found under {DATA_DIR.resolve()}")
        return

    # Get existing IDs from collection
    existing_ids: set[str] = set()
    for id in collection.get()["ids"]:
        existing_ids.add(id)

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
    docs: list[tuple[str | list[dict[str, str | None]], ...]] = []
    metas: list[dict[str, str]] = []
    ids: list[str] = []
    for i, (doc, meta, _id) in enumerate(new_items, 1):
        docs.append(doc)
        metas.append(meta)
        ids.append(_id)

        if len(docs) == BATCH or i == len(new_items):
            collection.add(
                documents=docs,  # pyright: ignore[reportArgumentType]
                metadatas=metas,  # pyright: ignore[reportArgumentType]
                ids=ids,
            )
            print(f"Indexed {len(ids)} / {len(new_items)} new chunks...")
            docs, metas, ids = [], [], []

    print(
        f"✅ Ingest complete. {collection.count()} total records in '{COLLECTION_NAME}'."
    )
