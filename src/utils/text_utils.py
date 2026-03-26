"""src/utils/text_utils.py — Gujarati text normalization helpers."""
from __future__ import annotations
import re
import hashlib
import unicodedata


def normalize_gujarati(text: str) -> str:
    """NFC normalization, remove HTML/URLs, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cc" or c == "\n")
    text = re.sub(r"[^\u0A00-\u0AFF\u0020-\u007E\n।,;:.!?0-9 ]", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def is_gujarati_text(text: str, threshold: float = 0.2) -> bool:
    """Returns True if at least `threshold` fraction of chars are Gujarati."""
    gu = sum(1 for c in text if "\u0A80" <= c <= "\u0AFF")
    return gu / max(len(text), 1) >= threshold


def is_valid_sentence(text: str, min_len: int = 10, max_len: int = 512) -> bool:
    return min_len <= len(text) <= max_len


def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
