import os
import json
import uuid
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1. Load Environment Variables
load_dotenv()
nest_asyncio.apply() # Fixes potential event loop issues with LlamaParse

# --- CONFIGURATION ---
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FILENAME = "Nvidia10K2025.pdf" # Make sure this matches your exact filename in data/raw/
OUTPUT_JSONL = os.path.join(PROCESSED_DIR, "corpus.jsonl")

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def parse_pdf_to_markdown(source_path, md_filename: str | None = None):
    """
    Uses LlamaParse to convert PDF -> Markdown.
    Checks if a .md file already exists to save API credits.

    If md_filename is set (e.g. alongside a per-corpus corpus.jsonl), writes there.
    """
    if md_filename is None:
        md_filename = source_path.replace(".pdf", ".md").replace(RAW_DIR, PROCESSED_DIR)

    os.makedirs(os.path.dirname(os.path.abspath(md_filename)), exist_ok=True)

    if os.path.exists(md_filename):
        print(f"✅ Found cached Markdown file: {md_filename}")
        with open(md_filename, "r", encoding="utf-8") as f:
            return f.read()

    print(f"⏳ Parsing {source_path} with LlamaParse (this may take a minute)...")
    
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        # fast_mode=True # Set to True if you want speed over table precision (not recommended for finance)
    )
    
    documents = parser.load_data(source_path)
    
    # LlamaParse returns a list of doc objects, usually one per page or combined.
    # We combine them into one massive markdown string.
    full_markdown_text = "\n\n".join([doc.text for doc in documents])
    
    # Save to disk for future runs
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(full_markdown_text)
    
    print(f"✅ Saved parsed Markdown to: {md_filename}")
    return full_markdown_text

def chunk_markdown_semantically(
    markdown_text,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 250,
    headers_to_split_on: list | None = None,
):
    """
    Phase 1: Split by Structure (Headers)
    Phase 2: Split by Size (character window)
    """
    print("✂️ Starting Semantic Chunking...")

    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    header_splits = markdown_splitter.split_text(markdown_text)

    print(f"   -> Split into {len(header_splits)} semantic sections based on headers.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    final_chunks = text_splitter.split_documents(header_splits)
    print(
        f"   -> Recursively split into {len(final_chunks)} final chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})."
    )
    return final_chunks

def save_to_jsonl(chunks, source_filename, output_jsonl: str | None = None):
    """
    Injects UUID, Source, and Metadata. Saves to JSONL.
    """
    out = output_jsonl or OUTPUT_JSONL
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(f"💾 Saving to {out}...")

    with open(out, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # Create Audit-Grade ID
            chunk_id = str(uuid.uuid4())
            
            # Flatten metadata into a readable string "Item 1 > Risk Factors"
            # The metadata keys come from the Header split (Header 1, Header 2...)
            metadata_path = " > ".join(chunk.metadata.values()) if chunk.metadata else "General"
            
            record = {
                "chunk_id": chunk_id,
                "source": source_filename,
                "section": metadata_path,
                "text": chunk.page_content,
                "metadata": chunk.metadata # Keep raw dict just in case
            }
            
            f.write(json.dumps(record) + "\n")
    
    print("✅ Pipeline Complete.")

def run_etl(
    pdf_filename: str | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    header_levels: list[tuple[str, str]] | None = None,
) -> bool:
    """
    Parse PDF in data/raw/, chunk, write data/processed/corpus.jsonl.
    Returns False if the PDF is missing.

    Optional chunking overrides for experiment presets (defaults: 1000 / 250).
    """
    filename = pdf_filename or FILENAME
    source_path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(source_path):
        print(f"❌ Error: File not found at {source_path}")
        return False

    cs = chunk_size if chunk_size is not None else 1000
    co = chunk_overlap if chunk_overlap is not None else 250

    corpus_jsonl = os.environ.get("RAG_CORPUS_JSONL", OUTPUT_JSONL)
    if not os.path.isabs(corpus_jsonl):
        # relative to cwd (repo root when running from project root)
        corpus_jsonl = os.path.abspath(corpus_jsonl)
    os.makedirs(os.path.dirname(corpus_jsonl), exist_ok=True)

    md_filename = os.path.join(
        os.path.dirname(corpus_jsonl),
        os.path.basename(source_path).replace(".pdf", ".md"),
    )

    raw_markdown = parse_pdf_to_markdown(source_path, md_filename=md_filename)
    final_chunks = chunk_markdown_semantically(
        raw_markdown,
        chunk_size=cs,
        chunk_overlap=co,
        headers_to_split_on=header_levels,
    )
    save_to_jsonl(final_chunks, filename, output_jsonl=corpus_jsonl)
    return True

def main():
    run_etl()


if __name__ == "__main__":
    main()
