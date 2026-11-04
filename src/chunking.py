"""
src/chunking.py
Text splitting.

chunk_pages() takes the output of loader.read_pdf() - a list of
(page_number, text) - and returns a list of {"page": int, "text": str}.

Chunks are built from whole sentences where possible, so they are not cut in
the middle of a sentence. A sentence that is longer than CHUNK_SIZE is split
at word boundaries. No chunk is ever longer than the chunk size.
"""

import re

# Embedding model all-MiniLM-L12-v2 reads at most 128 tokens (about 500-600
# characters). Keep CHUNK_SIZE at 500 or lower so no text is cut off.
CHUNK_SIZE = 500       # maximum characters per chunk
CHUNK_OVERLAP = 100    # characters repeated between neighbouring chunks

# Split after . ! ? when the next character starts a new sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text):
    """Split text into sentences (simple rule-based splitter)."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _hard_split(text, size, overlap):
    """Split one very long sentence into pieces of at most `size` characters."""
    pieces = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Prefer to cut at a space, but not too early in the window.
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always move forward
    return pieces


def _joined_len(parts):
    return len(" ".join(parts))


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split one string into overlapping chunks of at most `size` characters."""
    if size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0")
    if overlap < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative")
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    text = " ".join(text.split())  # collapse newlines and repeated spaces
    if not text:
        return []

    # 1. Break the text into pieces that each fit inside one chunk.
    pieces = []
    for sentence in split_sentences(text):
        if len(sentence) <= size:
            pieces.append(sentence)
        else:
            pieces.extend(_hard_split(sentence, size, overlap))

    # 2. Pack pieces into chunks, carrying a little text over as overlap.
    chunks = []
    current = []
    for piece in pieces:
        if current and _joined_len(current + [piece]) > size:
            chunks.append(" ".join(current))

            # Keep the last few pieces (up to `overlap` characters) for the next chunk.
            carry = []
            for prev in reversed(current):
                if _joined_len([prev] + carry) > overlap:
                    break
                carry.insert(0, prev)
            current = carry

            # The carried text plus the new piece must still fit.
            while current and _joined_len(current + [piece]) > size:
                current.pop(0)
        current.append(piece)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk every page and remember which page each chunk came from."""
    chunks = []
    for page_number, text in pages:
        for piece in chunk_text(text, size, overlap):
            chunks.append({"page": page_number, "text": piece})
    return chunks
