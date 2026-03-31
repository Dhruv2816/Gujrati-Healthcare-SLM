"""src/vectordb/chroma_client.py — ChromaDB ingestion & semantic search for medical books."""
from __future__ import annotations
import os
from typing import Optional
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.config import CHROMA_DIR, EMBED_MODEL_ID, CHROMA_COLLECTION

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False


def _extract_text_from_pdf(pdf_path: str, max_pages: int = 150) -> str:
    if not _FITZ_AVAILABLE:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
    doc = fitz.open(pdf_path)
    # Stop laptop crashing by reading only first 150 pages instead of 4000 pages
    lines = []
    for i, page in enumerate(doc):
        if i >= max_pages: 
            break
        lines.append(page.get_text())
    return "\n".join(lines)


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if len(chunk.strip()) > 50:  # skip tiny chunks
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


class ChromaClient:
    """ChromaDB client for the medical books vector collection."""

    def __init__(
        self,
        persist_dir: str = str(CHROMA_DIR),
        collection_name: str = CHROMA_COLLECTION,
        embed_model: str = EMBED_MODEL_ID,
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        # Force CPU to save scarce VRAM exclusively for the LLM
        self._embed_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model, device="cpu")
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── ingestion ────────────────────────────────────────
    def ingest_pdf(self, pdf_path: str, book_title: str = "") -> int:
        """Parse PDF → chunk → embed → upsert into ChromaDB. Returns chunk count."""
        text = _extract_text_from_pdf(pdf_path)
        chunks = _chunk_text(text)
        title = book_title or os.path.basename(pdf_path)

        ids  = [f"{title}_chunk_{i}" for i in range(len(chunks))]
        metas = [{"source": title, "chunk_idx": i} for i in range(len(chunks))]

        # Upsert in batches of 100
        batch = 100
        for start in range(0, len(chunks), batch):
            self._collection.upsert(
                ids=ids[start: start + batch],
                documents=chunks[start: start + batch],
                metadatas=metas[start: start + batch],
            )
        return len(chunks)

    def ingest_books_dir(self, books_dir: str = str(CHROMA_DIR.parent / "books")) -> dict:
        """Scan books_dir for PDFs and ingest all of them."""
        summary = {}
        for fname in os.listdir(books_dir):
            if fname.lower().endswith(".pdf"):
                path = os.path.join(books_dir, fname)
                count = self.ingest_pdf(path, book_title=fname)
                summary[fname] = count
                print(f"  ✅ {fname}: {count} chunks ingested")
        return summary

    # ── retrieval ────────────────────────────────────────
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search. Returns list of {text, source, score}."""
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count() or 1),
        )
        docs, metas, dists = (
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
        return [
            {
                "text":   doc,
                "source": meta.get("source", ""),
                "score":  round(1 - dist, 4),  # cosine similarity
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def count(self) -> int:
        return self._collection.count()

    def get_stats(self) -> dict:
        return {"collection": self._collection.name, "total_chunks": self.count()}
