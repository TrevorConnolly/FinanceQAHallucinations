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

def parse_pdf_to_markdown(source_path):
    """
    Uses LlamaParse to convert PDF -> Markdown.
    Checks if a .md file already exists to save API credits.
    """
    md_filename = source_path.replace(".pdf", ".md").replace(RAW_DIR, PROCESSED_DIR)
    
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

def chunk_markdown_semantically(markdown_text):
    """
    Phase 1: Split by Structure (Headers)
    Phase 2: Split by Size (Token limit)
    """
    print("✂️ Starting Semantic Chunking...")

    # 1. Split by Headers (The "Semantic" Part)
    # LlamaParse typically generates # for Title, ## for Sections, ### for Subsections
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = markdown_splitter.split_text(markdown_text)
    
    print(f"   -> Split into {len(header_splits)} semantic sections based on headers.")

    # 2. Split by Size (The "Recursive" Part)
    # Even a single section like "Risk Factors" might be 5000 tokens. We need to slice it down.
    # We keep the metadata from the headers attached!
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len, # Can use tiktoken here if preferred
        is_separator_regex=False,
    )
    
    final_chunks = text_splitter.split_documents(header_splits)
    print(f"   -> Recursively split into {len(final_chunks)} final chunks (max 1000 chars).")
    return final_chunks

def save_to_jsonl(chunks, source_filename):
    """
    Injects UUID, Source, and Metadata. Saves to JSONL.
    """
    print(f"💾 Saving to {OUTPUT_JSONL}...")
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
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

def main():
    source_path = os.path.join(RAW_DIR, FILENAME)
    
    if not os.path.exists(source_path):
        print(f"❌ Error: File not found at {source_path}")
        return

    # 1. Parse
    raw_markdown = parse_pdf_to_markdown(source_path)
    
    # 2. Chunk
    final_chunks = chunk_markdown_semantically(raw_markdown)
    
    # 3. Save
    save_to_jsonl(final_chunks, FILENAME)

if __name__ == "__main__":
    main()