
import os
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("."))

from src.vectordb.chroma_client import ChromaClient
from src.config import CHROMA_DIR, BOOKS_DIR

def reingest():
    print("--- 🗑️  WIPING OLD VECTOR DB ---")
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"   Deleted {CHROMA_DIR}")
    
    print("\n--- 📚 RE-INGESTING MEDICAL BOOKS (Skipping Front Matter) ---")
    client = ChromaClient()
    
    # Check if books exist
    if not os.path.exists(BOOKS_DIR):
        # Maybe check data/books
        BOOKS_DIR_ALT = Path("data/books")
        if os.path.exists(BOOKS_DIR_ALT):
            books_path = str(BOOKS_DIR_ALT)
        else:
            print(f"❌ Error: Books directory not found at {BOOKS_DIR}")
            return
    else:
        books_path = str(BOOKS_DIR)

    print(f"   Searching for PDFs in: {books_path}")
    summary = client.ingest_books_dir(books_path)
    
    print("\n--- ✅ RE-INGESTION COMPLETE ---")
    for book, count in summary.items():
        print(f"   - {book}: {count} relevant chunks")
    
    print(f"   Total Chunks in DB: {client.count()}")

if __name__ == "__main__":
    reingest()
